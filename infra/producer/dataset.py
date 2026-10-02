"""Parse NASA C-MAPSS turbofan files and shape them into the two published flows.

The train/test files are headerless, whitespace-separated rows with 26 fields:
unit_number, time_cycles, three operational settings, and 21 sensor
measurements (see data/readme.txt). Rows are split on arbitrary whitespace so
trailing spaces do not produce phantom fields.

Two projections come off each parsed observation:
- static_record: the batch/dimensional row uploaded to S3 as TSV (settings kept,
  sensors dropped).
- sensor_record: the wide sensor row published to Kinesis (via IoT Core), the
  composite key plus all 21 readings. The source rows have no native timestamp,
  so records carry none; time_cycles is the logical time axis.

The RUL_FD00x.txt files are handled separately: each holds one terminal-RUL value
per test engine instance (the true remaining cycles at the engine's last observed
cycle). They become rul_records, a small reference uploaded to its own S3 prefix,
and exist only for test engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SETTING_COLUMNS = ("setting_1", "setting_2", "setting_3")
SENSOR_IDS = tuple(f"sensor_{i}" for i in range(1, 22))
FIELD_COUNT = 2 + len(SETTING_COLUMNS) + len(SENSOR_IDS)  # 26

FAMILIES = ("FD001", "FD002", "FD003", "FD004")
SOURCES = ("train", "test")

STATIC_COLUMNS = (
    "engine_family",
    "source_flag",
    "unit_number",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
)

RUL_COLUMNS = ("engine_family", "source_flag", "unit_number", "terminal_rul")


@dataclass(frozen=True)
class Observation:
    """One row: a single operating cycle of one engine."""

    engine_family: str  # e.g. "FD001"
    source_flag: str  # "train" or "test"
    unit_number: int
    time_cycles: int
    settings: dict[str, float]  # setting_1..setting_3
    sensors: dict[str, float]  # sensor_1..sensor_21


def parse_line(line: str, engine_family: str, source_flag: str) -> Observation | None:
    """Parse one row. Returns None for blank lines."""
    fields = line.split()
    if not fields:
        return None
    if len(fields) != FIELD_COUNT:
        raise ValueError(f"expected {FIELD_COUNT} fields, got {len(fields)}: {line!r}")
    return Observation(
        engine_family=engine_family,
        source_flag=source_flag,
        unit_number=int(fields[0]),
        time_cycles=int(fields[1]),
        settings={name: float(fields[2 + i]) for i, name in enumerate(SETTING_COLUMNS)},
        sensors={name: float(fields[5 + i]) for i, name in enumerate(SENSOR_IDS)},
    )


def parse_file(path, engine_family: str, source_flag: str) -> list[Observation]:
    lines = Path(path).read_text().splitlines()
    parsed = (parse_line(line, engine_family, source_flag) for line in lines)
    return [obs for obs in parsed if obs is not None]


def discover_files(
    data_dir, families=FAMILIES, sources=SOURCES
) -> Iterator[tuple[str, str, Path]]:
    """Yield (engine_family, source_flag, path) for the C-MAPSS files present."""
    data_dir = Path(data_dir)
    for family in families:
        for source in sources:
            path = data_dir / f"{source}_{family}.txt"
            if path.exists():
                yield family, source, path


def load_observations(data_dir, families=FAMILIES, sources=SOURCES) -> list[Observation]:
    observations: list[Observation] = []
    for family, source, path in discover_files(data_dir, families, sources):
        observations.extend(parse_file(path, family, source))
    return observations


def static_record(obs: Observation) -> dict:
    """Dimensional metadata row (settings kept, sensors dropped)."""
    return {
        "engine_family": obs.engine_family,
        "source_flag": obs.source_flag,
        "unit_number": obs.unit_number,
        "time_cycles": obs.time_cycles,
        **obs.settings,
    }


def sensor_record(obs: Observation) -> dict:
    """One wide sensor row: the composite key plus all 21 sensor readings.

    Carries source_flag so that (engine_family, source_flag, unit_number,
    time_cycles) is a unique key in both flows — train and test reuse the same
    unit and cycle numbers, so without it the two flows cannot be joined back.
    """
    return {
        "engine_family": obs.engine_family,
        "source_flag": obs.source_flag,
        "unit_number": obs.unit_number,
        "time_cycles": obs.time_cycles,
        **obs.sensors,
    }


def rul_record(engine_family: str, unit_number: int, terminal_rul: int) -> dict:
    """One terminal-RUL label for a test engine instance."""
    return {
        "engine_family": engine_family,
        "source_flag": "test",
        "unit_number": unit_number,
        "terminal_rul": terminal_rul,
    }


def parse_rul_file(path, engine_family: str) -> list[dict]:
    """Parse a terminal-RUL file into label rows, in engine-instance order.

    Each line carries the RUL of one test engine at its last observed cycle. The
    values are ordered by unit, so the Nth non-blank line belongs to unit_number N.
    """
    labels: list[dict] = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 1:
            raise ValueError(f"expected 1 RUL value, got {len(fields)}: {line!r}")
        labels.append(rul_record(engine_family, len(labels) + 1, int(fields[0])))
    return labels


def discover_rul_files(data_dir, families=FAMILIES) -> Iterator[tuple[str, Path]]:
    """Yield (engine_family, path) for the RUL label files present."""
    data_dir = Path(data_dir)
    for family in families:
        path = data_dir / f"RUL_{family}.txt"
        if path.exists():
            yield family, path


def load_rul_labels(data_dir, families=FAMILIES) -> list[dict]:
    labels: list[dict] = []
    for family, path in discover_rul_files(data_dir, families):
        labels.extend(parse_rul_file(path, family))
    return labels
