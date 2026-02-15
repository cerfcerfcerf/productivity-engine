"""JSON adapter for normalized events."""

from __future__ import annotations

import json
from datetime import datetime

from productivity_engine.schema import NormalizedEvent

_REQUIRED_FIELDS = {"task_id", "timestamp", "action"}
_VALID_ACTIONS = {"created", "done", "snooze", "ignore"}


def _parse_item(item: dict, index: int) -> NormalizedEvent:
    missing = [field for field in _REQUIRED_FIELDS if not item.get(field)]
    if missing:
        raise ValueError(f"Item {index}: missing required fields {missing}")

    try:
        timestamp = datetime.fromisoformat(item["timestamp"])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Item {index}: malformed timestamp") from exc

    action = str(item["action"]).strip()
    if action not in _VALID_ACTIONS:
        raise ValueError(f"Item {index}: invalid action '{action}'")

    deadline_raw = item.get("deadline_hours")
    deadline_hours = None
    if deadline_raw is not None:
        try:
            deadline_hours = float(deadline_raw)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Item {index}: invalid deadline_hours") from exc

    category_raw = item.get("category")
    category = str(category_raw).strip() if category_raw else None

    return NormalizedEvent(
        task_id=str(item["task_id"]).strip(),
        timestamp=timestamp,
        action=action,
        deadline_hours=deadline_hours,
        category=category,
    )


def parse(file_path: str) -> list[NormalizedEvent]:
    """Parse JSON file into normalized events."""

    with open(file_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("JSON payload must be a list of objects")

    return [_parse_item(item, i) for i, item in enumerate(payload, start=1)]
