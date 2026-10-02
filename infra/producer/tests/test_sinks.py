import json

from dataset import RUL_COLUMNS, STATIC_COLUMNS, parse_line
from sinks import DRIFT_COLUMN, build_messages, chunks, prepare_chunks, prepare_files, put_objects, rows_to_tsv, shuffle_rows


def test_chunks_sizes():
    got = list(chunks(list(range(23)), 10))
    assert [len(c) for c in got] == [10, 10, 3]


def test_rows_to_tsv_header_and_delimiter():
    rows = [{col: col for col in STATIC_COLUMNS}]
    lines = rows_to_tsv(rows).splitlines()
    assert lines[0].split("\t") == list(STATIC_COLUMNS)
    assert lines[1].split("\t") == list(STATIC_COLUMNS)
    assert "\r" not in rows_to_tsv(rows)


def test_shuffle_deterministic_with_seed():
    rows = [{"unit_number": i} for i in range(50)]
    assert shuffle_rows(rows, seed=7) == shuffle_rows(rows, seed=7)
    assert shuffle_rows(rows, seed=1) != shuffle_rows(rows, seed=7)


class _FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body.decode()


def test_prepare_chunks_injects_schema_drift_in_trailing_chunks():
    rows = [{col: col for col in STATIC_COLUMNS} for _ in range(10)]
    items = prepare_chunks("engine_data/", rows, chunk_size=2, drift_fraction=0.5, seed=1)
    assert len(items) == 5  # 10 rows / 2 per chunk

    s3 = _FakeS3()
    put_objects(s3, "b", items)
    keys = [key for key, _ in items]
    assert set(s3.objects) == set(keys)

    first_header = s3.objects[keys[0]].splitlines()[0].split("\t")
    last_header = s3.objects[keys[-1]].splitlines()[0].split("\t")
    assert DRIFT_COLUMN not in first_header
    assert DRIFT_COLUMN in last_header


def test_prepare_chunks_no_drift_by_default():
    rows = [{col: col for col in STATIC_COLUMNS} for _ in range(4)]
    items = prepare_chunks("engine_data/", rows, chunk_size=2)
    for _, body in items:
        assert DRIFT_COLUMN not in body.decode().splitlines()[0]


def test_prepare_files_one_file_per_group_with_given_columns():
    groups = [
        ("RUL_FD001", [{col: col for col in RUL_COLUMNS} for _ in range(3)]),
        ("RUL_FD002", [{col: col for col in RUL_COLUMNS} for _ in range(2)]),
    ]
    items = prepare_files("engine_labels/", groups, columns=RUL_COLUMNS)
    assert len(items) == 2  # one unchunked file per fleet

    keys = [key for key, _ in items]
    assert len(set(keys)) == 2
    assert all(key.startswith("engine_labels/RUL_FD00") and key.endswith(".tsv") for key in keys)
    for _, body in items:
        header = body.decode().splitlines()[0].split("\t")
        assert header == list(RUL_COLUMNS)
        assert DRIFT_COLUMN not in header


def test_build_messages_packs_all_rows_into_json_arrays():
    sample = "1 12 " + " ".join(str(2 + i) for i in range(3 + 21))
    observations = [parse_line(sample, "FD001", "train") for _ in range(10)]
    messages = build_messages(observations, message_rows=4, seed=1)
    assert len(messages) == 3  # 10 rows / 4 per message

    rows = [row for message in messages for row in json.loads(message)]
    assert len(rows) == 10
    assert all(len(json.loads(message)) <= 4 for message in messages)
    assert set(rows[0]) == {"engine_family", "source_flag", "unit_number", "time_cycles"} | {
        f"sensor_{i}" for i in range(1, 22)
    }
