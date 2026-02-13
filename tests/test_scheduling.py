from datetime import datetime

from productivity_engine.features import extract_features
from productivity_engine.risk_model import train_logistic
from productivity_engine.schema import NormalizedEvent
from productivity_engine.scheduling import best_nudge_times


def test_best_nudge_times_returns_sorted_candidates():
    events = [
        NormalizedEvent("t1", datetime.fromisoformat("2025-01-01T08:00:00"), "created", 20, "work"),
        NormalizedEvent("t1", datetime.fromisoformat("2025-01-01T09:00:00"), "done", 19, "work"),
    ]
    features = extract_features(events)
    model = train_logistic(events)
    task = {"task_id": "t1", "deadline_hours": 10, "weekday": 3, "category": "work", "window": (8, 10)}
    result = best_nudge_times(task, features, model)
    assert 1 <= len(result) <= 2
    assert all(8 <= h <= 10 for h in result)
