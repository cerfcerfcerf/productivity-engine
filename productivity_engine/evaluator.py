"""Baseline vs adaptive evaluator."""

from __future__ import annotations


def compare(baseline_metrics: dict, adaptive_metrics: dict) -> dict:
    """Compare baseline and adaptive metric outcomes with percentage deltas."""

    base_completion = baseline_metrics.get("expected_completion_rate", baseline_metrics.get("completion_rate", 0.0))
    adapt_completion = adaptive_metrics.get("expected_completion_rate", adaptive_metrics.get("completion_rate", 0.0))

    base_delay = baseline_metrics.get("expected_delay_hours", baseline_metrics.get("avg_delay_hours", 0.0))
    adapt_delay = adaptive_metrics.get("expected_delay_hours", adaptive_metrics.get("avg_delay_hours", 0.0))

    base_ignore = baseline_metrics.get("expected_ignore_rate", baseline_metrics.get("ignore_rate", 0.0))
    adapt_ignore = adaptive_metrics.get("expected_ignore_rate", adaptive_metrics.get("ignore_rate", 0.0))

    def pct_change(old: float, new: float) -> float:
        if old == 0:
            return 0.0
        return ((new - old) / old) * 100.0

    return {
        "completion_improvement_pct": pct_change(base_completion, adapt_completion),
        "delay_reduction_pct": -pct_change(base_delay, adapt_delay),
        "ignore_reduction_pct": -pct_change(base_ignore, adapt_ignore),
    }
