"""Behavioral feature extraction."""

from __future__ import annotations

from collections import Counter, defaultdict

from productivity_engine.schema import NormalizedEvent


def extract_features(events: list[NormalizedEvent]) -> dict:
    """Extract aggregate behavioral features from normalized events."""

    completion_by_hour = Counter()
    ignore_by_hour = Counter()
    hour_total = Counter()

    by_category_done = Counter()
    by_category_total = Counter()

    task_events: dict[str, list[NormalizedEvent]] = defaultdict(list)

    done_count = 0
    for event in sorted(events, key=lambda e: e.timestamp):
        hour = event.timestamp.hour
        hour_total[hour] += 1
        task_events[event.task_id].append(event)

        if event.action == "done":
            completion_by_hour[hour] += 1
            done_count += 1
        elif event.action == "ignore":
            ignore_by_hour[hour] += 1

        if event.category:
            by_category_total[event.category] += 1
            if event.action == "done":
                by_category_done[event.category] += 1

    completion_rate_by_hour = {
        hour: completion_by_hour[hour] / hour_total[hour] for hour in range(24) if hour_total[hour]
    }
    ignore_rate_by_hour = {hour: ignore_by_hour[hour] / hour_total[hour] for hour in range(24) if hour_total[hour]}

    category_success_rate = {
        category: by_category_done[category] / total for category, total in by_category_total.items() if total
    }

    recent_streak = {}
    for task_id, entries in task_events.items():
        streak = 0
        for event in reversed(entries):
            if event.action == "done":
                streak += 1
            else:
                break
        recent_streak[task_id] = streak

    overall_completion_rate = done_count / len(events) if events else 0.0

    return {
        "completion_by_hour": completion_rate_by_hour,
        "ignore_by_hour": ignore_rate_by_hour,
        "category_success_rate": category_success_rate,
        "recent_streak": recent_streak,
        "overall_completion_rate": overall_completion_rate,
    }
