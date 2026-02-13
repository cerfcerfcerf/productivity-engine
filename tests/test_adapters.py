import json

import pytest

from productivity_engine.adapters.csv_adapter import parse as parse_csv
from productivity_engine.adapters.json_adapter import parse as parse_json


def test_csv_parse_success(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "task_id,timestamp,action,deadline_hours,category\n"
        "a,2025-01-01T09:00:00,created,12,work\n"
        "a,2025-01-01T10:00:00,done,11,work\n",
        encoding="utf-8",
    )
    events = parse_csv(str(path))
    assert len(events) == 2
    assert events[1].action == "done"


def test_csv_parse_invalid_row(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("task_id,timestamp,action\na,bad,done\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_csv(str(path))


def test_json_parse_success(tmp_path):
    path = tmp_path / "events.json"
    payload = [
        {"task_id": "a", "timestamp": "2025-01-01T09:00:00", "action": "created"},
        {"task_id": "a", "timestamp": "2025-01-01T10:00:00", "action": "done", "deadline_hours": 8},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    events = parse_json(str(path))
    assert len(events) == 2


def test_json_parse_malformed(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps([{"task_id": "a", "timestamp": "bad", "action": "done"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        parse_json(str(path))
