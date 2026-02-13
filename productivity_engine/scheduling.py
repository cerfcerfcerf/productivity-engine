"""Adaptive reminder schedule selection."""

from __future__ import annotations

from productivity_engine.risk_model import compute_risk


def best_nudge_times(task: dict, features: dict, model) -> list[int]:
    """Choose one or two lowest-risk hours within task window."""

    start, end = task.get("window", (9, 17))
    candidates = list(range(int(start), int(end) + 1))
    ranked = sorted(candidates, key=lambda hour: (compute_risk(task, features, model, hour), hour))
    if len(ranked) <= 1:
        return ranked
    return ranked[:2]
