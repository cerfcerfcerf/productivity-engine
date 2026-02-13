from datetime import datetime

from productivity_engine.features import extract_features
from productivity_engine.risk_model import compute_risk, heuristic_risk, train_logistic
from productivity_engine.schema import NormalizedEvent


def sample_events():
    return [
        NormalizedEvent("t1", datetime.fromisoformat("2025-01-01T09:00:00"), "created", 10, "work"),
        NormalizedEvent("t1", datetime.fromisoformat("2025-01-01T10:00:00"), "done", 9, "work"),
        NormalizedEvent("t2", datetime.fromisoformat("2025-01-02T11:00:00"), "created", 5, "health"),
        NormalizedEvent("t2", datetime.fromisoformat("2025-01-02T12:00:00"), "ignore", 4, "health"),
    ]


def test_heuristic_risk_bounds():
    features = extract_features(sample_events())
    score = heuristic_risk({"task_id": "t1", "deadline_hours": 4}, features, hour=10)
    assert 0.0 <= score <= 1.0


def test_logistic_training_and_compute_risk():
    events = sample_events()
    model = train_logistic(events)
    features = extract_features(events)
    task = {"task_id": "t1", "deadline_hours": 12, "weekday": 2, "category": "work"}
    risk = compute_risk(task, features, model, hour=9)
    assert 0.0 <= risk <= 1.0
