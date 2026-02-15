"""Core data schema for productivity events."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class NormalizedEvent:
    """Normalized event record used by all modules."""

    task_id: str
    timestamp: datetime
    action: str
    deadline_hours: Optional[float]
    category: Optional[str]
