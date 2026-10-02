import pytest

from dataset import (
    FIELD_COUNT,
    RUL_COLUMNS,
    SENSOR_IDS,
    STATIC_COLUMNS,
    parse_file,
    parse_line,
    parse_rul_file,
    rul_record,
    sensor_record,
    static_record,
)

# A well-formed row: unit 1, cycle 12, 3 settings, then 21 sensors.
SAMPLE = "1 12 " + " ".join(str(2 + i) for i in range(3 + 21))
OBS = parse_line(SAMPLE, "FD001", "train")


def test_parse_line_fields():
    obs = parse_line(SAMPLE, "FD001", "train")
    assert obs.engine_family == "FD001"
    assert obs.source_flag == "train"
    assert obs.unit_number == 1
    assert obs.time_cycles == 12
    assert obs.settings == {"setting_1": 2.0, "setting_2": 3.0, "setting_3": 4.0}
    assert len(obs.sensors) == 21
    assert set(obs.sensors) == set(SENSOR_IDS)
    assert obs.sensors["sensor_1"] == 5.0


def test_parse_line_tolerates_trailing_whitespace():
    assert parse_line(SAMPLE + "   ", "FD001", "train") is not None


def test_parse_line_blank_is_none():
    assert parse_line("   ", "FD001", "train") is None


def test_parse_line_wrong_field_count_raises():
    with pytest.raises(ValueError):
        parse_line("1 2 3", "FD001", "train")


def test_parse_file(tmp_path):
    f = tmp_path / "train_FD001.txt"
    f.write_text(SAMPLE + "\n" + SAMPLE + "\n\n")
    assert len(parse_file(f, "FD001", "train")) == 2


def test_field_count_is_26():
    assert FIELD_COUNT == 26


def test_static_record_columns():
    record = static_record(OBS)
    assert set(record) == set(STATIC_COLUMNS)
    assert record["engine_family"] == "FD001"
    assert record["source_flag"] == "train"
    assert record["setting_3"] == 4.0
    assert "sensor_1" not in record


def test_sensor_record_wide_key_and_sensors_no_timestamp():
    record = sensor_record(OBS)
    assert set(record) == {"engine_family", "source_flag", "unit_number", "time_cycles", *SENSOR_IDS}
    assert record["engine_family"] == "FD001"
    assert record["source_flag"] == "train"
    assert record["unit_number"] == 1
    assert record["time_cycles"] == 12
    assert record["sensor_1"] == 5.0
    assert "timestamp" not in record


def test_rul_record_columns():
    record = rul_record("FD002", 5, 77)
    assert set(record) == set(RUL_COLUMNS)
    assert record == {"engine_family": "FD002", "source_flag": "test", "unit_number": 5, "terminal_rul": 77}


def test_parse_rul_file_units_follow_line_order(tmp_path):
    # Trailing spaces mimic the real RUL_FD00x.txt files.
    f = tmp_path / "RUL_FD001.txt"
    f.write_text("112 \n98\n69\n")
    labels = parse_rul_file(f, "FD001")
    assert [row["unit_number"] for row in labels] == [1, 2, 3]
    assert [row["terminal_rul"] for row in labels] == [112, 98, 69]
    assert all(row["engine_family"] == "FD001" and row["source_flag"] == "test" for row in labels)


def test_parse_rul_file_skips_blank_lines(tmp_path):
    f = tmp_path / "RUL_FD001.txt"
    f.write_text("112\n\n98\n\n")
    labels = parse_rul_file(f, "FD001")
    assert [row["unit_number"] for row in labels] == [1, 2]


def test_parse_rul_file_multi_value_line_raises(tmp_path):
    f = tmp_path / "RUL_FD001.txt"
    f.write_text("112 98\n")
    with pytest.raises(ValueError):
        parse_rul_file(f, "FD001")
