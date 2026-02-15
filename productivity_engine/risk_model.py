"""Hybrid heuristic + logistic risk model."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from productivity_engine.schema import NormalizedEvent


@dataclass
class TrainedRiskModel:
    """Container for deterministic logistic-like model parameters."""

    feature_names: list[str]
    weights: list[float]


def _deadline_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 6:
        return "lt6"
    if value < 24:
        return "6to24"
    if value < 72:
        return "24to72"
    return "gte72"


def _vectorize(row: dict, feature_names: list[str]) -> list[float]:
    row_map = {
        "hour": float(row["hour"]),
        "weekday": float(row["weekday"]),
        f"deadline_bucket={row['deadline_bucket']}": 1.0,
        f"category={row['category']}": 1.0,
    }
    return [row_map.get(name, 0.0) for name in feature_names]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def heuristic_risk(task: dict, features: dict, hour: int, return_components: bool = False):
    """Compute a bounded heuristic risk score between 0 and 1."""

    completion_rate = features.get("completion_by_hour", {}).get(hour, features.get("overall_completion_rate", 0.0))
    ignore_rate = features.get("ignore_by_hour", {}).get(hour, 0.0)
    deadline_hours = task.get("deadline_hours")
    streak = features.get("recent_streak", {}).get(task.get("task_id", ""), 0)

    hour_penalty = 0.4 * (1.0 - completion_rate)
    urgency = 0.0 if deadline_hours is None else max(0.0, (24.0 - deadline_hours) / 24.0)
    ignore_penalty = 0.3 * ignore_rate
    deadline_urgency = 0.25 * urgency
    streak_bonus = 0.2 * min(0.3, streak * 0.05)

    score = hour_penalty + ignore_penalty + deadline_urgency - streak_bonus
    bounded = max(0.0, min(1.0, score))

    if return_components:
        return {
            "score": bounded,
            "hour_penalty": hour_penalty,
            "ignore_penalty": ignore_penalty,
            "deadline_urgency": deadline_urgency,
            "streak_bonus": streak_bonus,
            "base_rate": completion_rate,
        }

    return bounded


def train_logistic(events: list[NormalizedEvent]) -> TrainedRiskModel:
    """Train a deterministic logistic classifier from event records."""

    rows = []
    y = []
    for event in events:
        rows.append(
            {
                "hour": event.timestamp.hour,
                "weekday": event.timestamp.weekday(),
                "deadline_bucket": _deadline_bucket(event.deadline_hours),
                "category": event.category or "unknown",
            }
        )
        y.append(1.0 if event.action == "done" else 0.0)

    feature_names = ["bias", "hour", "weekday"]
    feature_names.extend(sorted({f"deadline_bucket={r['deadline_bucket']}" for r in rows} or {"deadline_bucket=unknown"}))
    feature_names.extend(sorted({f"category={r['category']}" for r in rows} or {"category=unknown"}))

    x = [[1.0, *_vectorize(r, feature_names[1:])] for r in rows] if rows else [[1.0] + [0.0] * (len(feature_names) - 1)]
    y_train = y if y else [0.5]

    weights = [0.0] * len(feature_names)
    lr = 0.01
    for _ in range(400):
        grads = [0.0] * len(weights)
        for row, label in zip(x, y_train):
            pred = _sigmoid(sum(w * v for w, v in zip(weights, row)))
            err = pred - label
            for i, value in enumerate(row):
                grads[i] += err * value
        n = len(x)
        for i in range(len(weights)):
            weights[i] -= lr * grads[i] / n

    return TrainedRiskModel(feature_names=feature_names, weights=weights)


def logistic_risk(model: TrainedRiskModel, task: dict, hour: int) -> float:
    """Estimate risk from trained logistic model."""

    row = {
        "hour": hour,
        "weekday": int(task.get("weekday", 0)),
        "deadline_bucket": _deadline_bucket(task.get("deadline_hours")),
        "category": task.get("category") or "unknown",
    }
    values = [1.0, *_vectorize(row, model.feature_names[1:])]
    probability_done = _sigmoid(sum(w * v for w, v in zip(model.weights, values)))
    return 1.0 - float(probability_done)


def compute_risk(task: dict, features: dict, model: TrainedRiskModel, hour: int) -> float:
    """Compute final hybrid risk score."""

    h = heuristic_risk(task, features, hour)
    l = logistic_risk(model, task, hour)
    return max(0.0, min(1.0, 0.5 * h + 0.5 * l))
