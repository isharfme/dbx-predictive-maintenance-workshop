"""Publish the two workshop data flows into the deployed AWS infra, with the
small AWS-client glue each flow needs.

- The static flow blends, shuffles and chunks the dimensional rows into TSVs
  under an S3 prefix, so Auto Loader can ingest them incrementally.
- The stream flow shuffles the wide sensor rows and packs them into JSON-array
  messages published to AWS IoT Core over MQTT; the deployed IoT topic rule
  routes each message on into Kinesis (we never write Kinesis directly). The
  MQTT SDK is imported lazily so the static path stays light.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import secrets
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import boto3

from dataset import STATIC_COLUMNS, Observation, sensor_record, static_record

ROOT_CA_URL = "https://www.amazontrust.com/repository/AmazonRootCA1.pem"

# Cold on-demand Kinesis starts at roughly 4,000 records/s and 4 MB/s for writes
# and only scales up over minutes, so a one-shot run is capped near that. Publishing
# under both ceilings keeps the IoT rule's PutRecord from tripping throughput
# throttling, which would silently drop records (the rule has no error action).
MAX_MESSAGES_PER_SEC = 3500
MAX_BYTES_PER_SEC = 3_500_000

# A late-arriving extra column, added to the last chunks of a run, so the demo
# can show Auto Loader coping with a schema change (rescued data / evolution).
DRIFT_COLUMN = "maintenance_flag"
DRIFT_VALUES = ("ok", "inspect", "due")


# --- AWS clients -----------------------------------------------------------


def s3_client(sess: boto3.Session):
    return sess.client("s3")


def iot_data_endpoint(sess: boto3.Session) -> str:
    """The account/region-wide IoT Core data endpoint (iot:Data-ATS)."""
    return sess.client("iot").describe_endpoint(endpointType="iot:Data-ATS")[
        "endpointAddress"
    ]


def ensure_root_ca(path) -> str:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ROOT_CA_URL, path)
    return str(path)


# --- Static flow: chunked TSV to S3 ----------------------------------------


def build_rows(observations: Iterable[Observation]) -> list[dict]:
    return [static_record(obs) for obs in observations]


def shuffle_rows(rows: Sequence[dict], seed: int | None = None) -> list[dict]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def chunks(rows: Sequence[dict], size: int) -> Iterator[list[dict]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def rows_to_tsv(rows: Iterable[dict], columns=STATIC_COLUMNS) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(columns), delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def prepare_chunks(
    prefix: str,
    rows: Sequence[dict],
    chunk_size: int,
    drift_fraction: float = 0.0,
    seed: int | None = None,
) -> list[tuple[str, bytes]]:
    """Build the (object key, TSV body) pair for each chunk, ready to upload.

    The last `drift_fraction` of chunks gain an extra `maintenance_flag` column,
    so a downstream Auto Loader run meets a schema change partway through.
    """
    total = math.ceil(len(rows) / chunk_size) if rows else 0
    drift_from = total - math.ceil(total * drift_fraction) + 1 if drift_fraction else total + 1
    rng = random.Random(seed)
    # A per-run token in every filename, so re-running the producer lands new object
    # keys instead of overwriting the last run's. Auto Loader tracks files by path and
    # skips same-named overwrites, so unique names are what let each run re-ingest
    # incrementally without first clearing the notebook's checkpoint.
    run_token = secrets.token_hex(4)
    items: list[tuple[str, bytes]] = []
    for n, chunk in enumerate(chunks(rows, chunk_size), start=1):
        if n >= drift_from:
            chunk_columns = STATIC_COLUMNS + (DRIFT_COLUMN,)
            chunk = [{**row, DRIFT_COLUMN: rng.choice(DRIFT_VALUES)} for row in chunk]
        else:
            chunk_columns = STATIC_COLUMNS
        key = f"{prefix}part-{run_token}-{n:04d}.tsv"
        items.append((key, rows_to_tsv(chunk, chunk_columns).encode()))
    return items


def prepare_files(
    prefix: str,
    groups: Sequence[tuple[str, Sequence[dict]]],
    columns: Sequence[str] = STATIC_COLUMNS,
) -> list[tuple[str, bytes]]:
    """Build one (object key, TSV body) pair per named group, unchunked.

    Each key carries a per-run token, like the chunked flow, so a re-run lands new
    objects instead of overwriting the last run's.
    """
    run_token = secrets.token_hex(4)
    return [
        (f"{prefix}{name}-{run_token}.tsv", rows_to_tsv(rows, columns).encode())
        for name, rows in groups
    ]


def put_objects(
    s3,
    bucket: str,
    items: Sequence[tuple[str, bytes]],
    workers: int = 16,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Upload prepared (key, body) pairs to S3 concurrently.

    The uploads are independent network round-trips, so a small thread pool turns
    a run of sequential PUTs into one that finishes in a few seconds.
    """
    total = len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(s3.put_object, Bucket=bucket, Key=key, Body=body) for key, body in items]
        for future in as_completed(futures):
            future.result()
            done += 1
            if on_progress is not None:
                on_progress(done, total)


def reset_prefix(s3, bucket: str, prefix: str) -> int:
    """Delete every object under prefix. Returns the count removed."""
    paginator = s3.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            deleted += len(objects)
    return deleted


# --- Stream flow: batched sensor messages to IoT Core over MQTT ------------


def build_messages(
    observations: Iterable[Observation], message_rows: int, seed: int | None = None
) -> list[str]:
    """Shuffle the wide sensor rows and pack them into JSON-array messages.

    Each message is a batch of up to `message_rows` independent, fully-keyed
    rows. Shuffling on every run varies arrival order across runs and spreads the
    resulting records across Kinesis shards.
    """
    rows = shuffle_rows([sensor_record(obs) for obs in observations], seed=seed)
    return [json.dumps(chunk) for chunk in chunks(rows, message_rows)]


def connect(endpoint: str, cert: str, key: str, ca: str, client_id: str):
    from awsiot import mqtt_connection_builder

    connection = mqtt_connection_builder.mtls_from_path(
        endpoint=endpoint,
        cert_filepath=cert,
        pri_key_filepath=key,
        ca_filepath=ca,
        client_id=client_id,
        clean_session=True,
        keep_alive_secs=30,
    )
    connection.connect().result()
    return connection


def publish(
    connection,
    messages: Sequence[str],
    topic: str,
    qos: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
    max_messages_per_sec: int = MAX_MESSAGES_PER_SEC,
    max_bytes_per_sec: int = MAX_BYTES_PER_SEC,
) -> int:
    """Publish array-messages to `topic`, paced under the Kinesis write ceilings.

    Pacing is cumulative: after each publish we hold the average message and byte
    rates below the caps by sleeping when the run is running ahead. At QoS 0 there
    are no acknowledgements to await, so the rate has no ack bottleneck and no
    unacked backlog to overrun IoT Core's in-flight limit; delivery is best-effort.
    The final packet's write is awaited so a disconnect can't cut it off. Returns
    the count published.
    """
    from awscrt import mqtt

    level = mqtt.QoS(qos)
    total = len(messages)
    sent = 0
    bytes_sent = 0
    last = None
    start = time.monotonic()
    for payload in messages:
        body = payload.encode()
        last, _ = connection.publish(topic=topic, payload=body, qos=level)
        sent += 1
        bytes_sent += len(body)
        if on_progress is not None:
            on_progress(sent, total)
        target = max(sent / max_messages_per_sec, bytes_sent / max_bytes_per_sec)
        ahead = target - (time.monotonic() - start)
        if ahead > 0:
            time.sleep(ahead)
    if last is not None:
        last.result()
    return sent
