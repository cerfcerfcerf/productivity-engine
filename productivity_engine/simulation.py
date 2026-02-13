"""Deterministic simulation routines."""

from __future__ import annotations

from productivity_engine.features import extract_features
from productivity_engine.metrics import compute_metrics
from productivity_engine.risk_model import compute_risk, train_logistic
from productivity_engine.schema import NormalizedEvent


def simulate_baseline(events: list[NormalizedEvent]) -> dict:
    """Simulate baseline outcomes using historical rates only."""

    metrics = compute_metrics(events)
    return {
        "expected_completion_rate": metrics["completion_rate"],
        "expected_ignore_rate": metrics["ignore_rate"],
        "expected_delay_hours": metrics["avg_delay_hours"],
    }


def simulate_adaptive(events: list[NormalizedEvent]) -> dict:
    """Simulate adaptive outcomes from hybrid risk model over historical tasks."""

    features = extract_features(events)
    model = train_logistic(events)

    unique_tasks = {}
    for event in events:
        if event.task_id not in unique_tasks and event.action == "created":
            unique_tasks[event.task_id] = {
                "task_id": event.task_id,
                "deadline_hours": event.deadline_hours,
                "category": event.category,
                "weekday": event.timestamp.weekday(),
            }

    if not unique_tasks:
        return {
            "expected_completion_rate": 0.0,
            "expected_ignore_rate": 0.0,
            "expected_delay_hours": 0.0,
        }

    risks = [compute_risk(task, features, model, hour=10) for task in unique_tasks.values()]
    avg_risk = sum(risks) / len(risks)

    baseline = simulate_baseline(events)
    improvement = max(0.0, 0.15 * (1 - avg_risk))
    return {
        "expected_completion_rate": min(1.0, baseline["expected_completion_rate"] + improvement),
        "expected_ignore_rate": max(0.0, baseline["expected_ignore_rate"] - 0.5 * improvement),
        "expected_delay_hours": max(0.0, baseline["expected_delay_hours"] * (1 - 0.4 * improvement)),
    }
