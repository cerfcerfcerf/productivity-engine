"""Productivity outcome metrics."""

from __future__ import annotations

from collections import defaultdict

from productivity_engine.schema import NormalizedEvent


def compute_metrics(events: list[NormalizedEvent]) -> dict:
    """Compute completion, ignore, delay and task count metrics."""

    if not events:
        return {
            "completion_rate": 0.0,
            "ignore_rate": 0.0,
            "avg_delay_hours": 0.0,
            "total_tasks": 0,
        }

    by_task = defaultdict(list)
    done_count = 0
    ignore_count = 0
    for event in events:
        by_task[event.task_id].append(event)
        done_count += 1 if event.action == "done" else 0
        ignore_count += 1 if event.action == "ignore" else 0

    delays = []
    for task_id, task_events in by_task.items():
        created = min((e.timestamp for e in task_events if e.action == "created"), default=None)
        done = min((e.timestamp for e in task_events if e.action == "done"), default=None)
        if created and done and done >= created:
            delays.append((done - created).total_seconds() / 3600.0)

    return {
        "completion_rate": done_count / len(events),
        "ignore_rate": ignore_count / len(events),
        "avg_delay_hours": sum(delays) / len(delays) if delays else 0.0,
        "total_tasks": len(by_task),
    }
