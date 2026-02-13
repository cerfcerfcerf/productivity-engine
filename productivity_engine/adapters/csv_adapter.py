"""CSV adapter for normalized events."""

from __future__ import annotations

import csv
from datetime import datetime

from productivity_engine.schema import NormalizedEvent

_REQUIRED_FIELDS = {"task_id", "timestamp", "action"}
_VALID_ACTIONS = {"created", "done", "snooze", "ignore"}


def _parse_row(row: dict, row_number: int) -> NormalizedEvent:
    missing = [field for field in _REQUIRED_FIELDS if not row.get(field)]
    if missing:
        raise ValueError(f"Row {row_number}: missing required fields {missing}")

    try:
        timestamp = datetime.fromisoformat(row["timestamp"])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Row {row_number}: malformed timestamp") from exc

    action = row["action"].strip()
    if action not in _VALID_ACTIONS:
        raise ValueError(f"Row {row_number}: invalid action '{action}'")

    deadline_raw = row.get("deadline_hours")
    deadline_hours = None
    if deadline_raw not in (None, ""):
        try:
            deadline_hours = float(deadline_raw)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Row {row_number}: invalid deadline_hours") from exc

    category_raw = row.get("category")
    category = category_raw.strip() if category_raw else None

    return NormalizedEvent(
        task_id=row["task_id"].strip(),
        timestamp=timestamp,
        action=action,
        deadline_hours=deadline_hours,
        category=category,
    )


def parse(file_path: str) -> list[NormalizedEvent]:
    """Parse CSV file into a list of normalized events."""

    with open(file_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []

        events: list[NormalizedEvent] = []
        for row_number, row in enumerate(reader, start=2):
            events.append(_parse_row(row, row_number))
        return events
