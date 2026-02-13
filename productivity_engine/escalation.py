"""Deadline-based escalation rules."""

from __future__ import annotations


def deadline_schedule(hours_to_deadline: float) -> list[float]:
    """Return relative reminder offsets based on time-to-deadline."""

    if hours_to_deadline >= 72:
        return [24]
    if 24 <= hours_to_deadline < 72:
        return [12, 24]
    if 6 <= hours_to_deadline < 24:
        return [float(v) for v in range(3, int(hours_to_deadline) + 1, 3)]

    max_count = min(5, max(1, int(hours_to_deadline)))
    return [float(v) for v in range(1, max_count + 1)]
