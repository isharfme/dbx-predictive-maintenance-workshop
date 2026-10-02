"""Workshop data producer.

Local utility that reads the repo's C-MAPSS files and drives the flows into the
deployed AWS infra:

  file    write the settings TSVs (chunked, with schema drift) and one terminal-RUL
          reference TSV per fleet to their S3 prefixes
  stream  publish batched sensor messages to IoT Core (routed to Kinesis by the rule)
  reset   empty the static-metadata prefix so a run can start clean

AWS profile and region come from the repo-root .env (AWS_PROFILE, AWS_REGION);
every other setting is a fixed default. --batches is the one knob: it splits a run
into N batches and pauses between them so you can narrate the flow.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

import dataset
import sinks

REPO_ROOT = Path(__file__).resolve().parents[2]
CERTS_DIR = Path(__file__).resolve().parents[1] / "aws" / "certs"
ENV_FILE = REPO_ROOT / ".env"

DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_STATIC_PREFIX = "engine_data/"
DEFAULT_LABELS_PREFIX = "engine_labels/"
DEFAULT_STREAM_TOPIC = "pdm-ws/sensors/stream"
DEFAULT_CLIENT_ID = "pdm-ws-producer"
DEFAULT_CERT = CERTS_DIR / "pdm-ws-producer.cert.pem"
DEFAULT_KEY = CERTS_DIR / "pdm-ws-producer.private.key"
DEFAULT_CA = CERTS_DIR / "AmazonRootCA1.pem"
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_DRIFT_FRACTION = 0.2
# Rows per stream message: ~187 packs the shuffled fleet into ~1,400 messages,
# each well under the 128 KB IoT Core payload limit.
DEFAULT_MESSAGE_ROWS = 187
# QoS 0: best-effort delivery with no per-message acknowledgement, so publishing
# has no ack bottleneck and can't overrun IoT Core's in-flight limit. The Silver
# notebook de-duplicates, and pacing (below) keeps us under the write ceilings.
DEFAULT_QOS = 0


def _session() -> boto3.Session:
    """A boto3 session from the .env AWS profile and region, or a clear error."""
    profile = os.environ.get("AWS_PROFILE")
    region = os.environ.get("AWS_REGION")
    missing = [n for n, v in (("AWS_PROFILE", profile), ("AWS_REGION", region)) if not v]
    if missing:
        raise SystemExit(
            "cli.py: missing " + ", ".join(missing)
            + "; set it in .env at the repo root (copy .env.sample)"
        )
    return boto3.Session(profile_name=profile, region_name=region)


def _default_data_bucket() -> str:
    """The static-metadata bucket name, derived from .env the same way deploy.sh
    does: ${PREFIX:-pdm-ws}-data${BUCKET_SUFFIX:+-$BUCKET_SUFFIX}."""
    prefix = os.environ.get("PREFIX") or "pdm-ws"
    suffix = os.environ.get("BUCKET_SUFFIX")
    return f"{prefix}-data" + (f"-{suffix}" if suffix else "")


def _load():
    return dataset.load_observations(DEFAULT_DATA_DIR)


def _batch_prompt(done_desc: str) -> bool:
    """Prompt at a batch boundary. Return True to keep batching (next batch),
    False to switch batching off and emit the rest in one go."""
    reply = input(
        f"  paused after {done_desc} — press N then Enter for the next batch, "
        "or just Enter to send the rest: "
    )
    return reply.strip() != ""


def _split_batches(items, n):
    """Split items into n contiguous batches (the last carries any remainder);
    n unset means a single batch, so the whole run goes out without pausing."""
    if not n or n < 1:
        return [list(items)]
    size = math.ceil(len(items) / n)
    return [items[i : i + size] for i in range(0, len(items), size)]


class _Progress:
    """A single overwriting stderr line: done/total, throughput, elapsed.

    `paused` accumulates time spent waiting at batch prompts so the reported
    elapsed and rate reflect only active sending, not how long you narrated.
    """

    def __init__(self, noun: str, total: int):
        self.noun = noun
        self.total = total
        self.start = time.monotonic()
        self.paused = 0.0

    def update(self, done: int):
        elapsed = time.monotonic() - self.start - self.paused
        rate = done / elapsed if elapsed > 0 else 0.0
        print(
            f"\r  {done}/{self.total} {self.noun}  {rate:,.0f}/s  {elapsed:,.1f}s",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self):
        print(file=sys.stderr)


def _put_in_batches(s3, bucket: str, items, batches_arg):
    """Upload prepared (key, body) items to S3, pausing between batches if asked."""
    total = len(items)
    batches = _split_batches(items, batches_arg)
    progress = _Progress("files", total)
    pausing = batches_arg is not None
    sent = 0
    for i, batch in enumerate(batches, start=1):
        sinks.put_objects(s3, bucket, batch, on_progress=lambda done, _n, base=sent: progress.update(base + done))
        sent += len(batch)
        if pausing and i < len(batches):
            paused_at = time.monotonic()
            keep = _batch_prompt(f"{sent}/{total} file(s)")
            progress.paused += time.monotonic() - paused_at
            if not keep:
                pausing = False
    progress.close()


def cmd_file(args):
    bucket = _default_data_bucket()
    s3 = sinks.s3_client(_session())

    # Settings: blend + shuffle the dimensional rows into chunked TSVs, with a late schema
    # drift in the trailing chunks for the Auto Loader part of the demo.
    rows = sinks.shuffle_rows(sinks.build_rows(_load()), seed=None)
    items = sinks.prepare_chunks(
        DEFAULT_STATIC_PREFIX, rows, DEFAULT_CHUNK_SIZE, drift_fraction=DEFAULT_DRIFT_FRACTION, seed=None
    )

    # Labels: one small terminal-RUL TSV per fleet (unchunked), so the published output
    # mirrors the source RUL_FD00x.txt files instead of collapsing into one.
    label_groups = [
        (f"RUL_{family}", dataset.parse_rul_file(path, family))
        for family, path in dataset.discover_rul_files(DEFAULT_DATA_DIR)
    ]
    label_count = sum(len(group) for _, group in label_groups)
    items += sinks.prepare_files(DEFAULT_LABELS_PREFIX, label_groups, columns=dataset.RUL_COLUMNS)

    _put_in_batches(s3, bucket, items, args.batches)
    print(
        f"done: {len(items)} file(s) to s3://{bucket}/ "
        f"({len(rows)} setting row(s) under {DEFAULT_STATIC_PREFIX}; "
        f"{label_count} label(s) in {len(label_groups)} fleet file(s) under {DEFAULT_LABELS_PREFIX})"
    )


def cmd_stream(args):
    sess = _session()
    endpoint = sinks.iot_data_endpoint(sess)
    ca = sinks.ensure_root_ca(DEFAULT_CA)
    messages = sinks.build_messages(_load(), DEFAULT_MESSAGE_ROWS, seed=None)
    total = len(messages)
    batches = _split_batches(messages, args.batches)
    connection = sinks.connect(
        endpoint, str(DEFAULT_CERT), str(DEFAULT_KEY), ca, DEFAULT_CLIENT_ID
    )
    progress = _Progress("messages", total)
    pausing = args.batches is not None
    sent = 0
    try:
        for i, batch in enumerate(batches, start=1):
            sinks.publish(
                connection,
                batch,
                DEFAULT_STREAM_TOPIC,
                DEFAULT_QOS,
                on_progress=lambda done, _n, base=sent: progress.update(base + done),
            )
            sent += len(batch)
            if pausing and i < len(batches):
                paused_at = time.monotonic()
                keep = _batch_prompt(f"{sent}/{total} message(s)")
                progress.paused += time.monotonic() - paused_at
                if not keep:
                    pausing = False
    finally:
        connection.disconnect().result()
    progress.close()
    print(f"published {sent} message(s) to {DEFAULT_STREAM_TOPIC} via {endpoint}")


def cmd_reset(args):
    bucket = _default_data_bucket()
    s3 = sinks.s3_client(_session())
    removed = sinks.reset_prefix(s3, bucket, DEFAULT_STATIC_PREFIX)
    print(f"deleted {removed} object(s) under s3://{bucket}/{DEFAULT_STATIC_PREFIX}")
    print("Kinesis records age out via the stream's retention; not purged here.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=None,
        metavar="N",
        help="split the run into N batches (files for the file command, messages for stream) "
        "and pause between them so you can narrate the flow; press N then Enter for the next "
        "batch, or just Enter to send the rest. Unset sends everything in one shot",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("file", help="write settings TSVs and per-fleet RUL TSVs to S3").set_defaults(func=cmd_file)
    sub.add_parser("stream", help="publish batched sensor messages to IoT Core").set_defaults(func=cmd_stream)
    sub.add_parser("reset", help="empty the static-metadata prefix").set_defaults(func=cmd_reset)

    return parser


def main(argv=None):
    load_dotenv(ENV_FILE)
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
