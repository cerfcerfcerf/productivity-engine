"""Demo script for productivity-engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from productivity_engine.adapters.csv_adapter import parse
from productivity_engine.evaluator import compare
from productivity_engine.simulation import simulate_adaptive, simulate_baseline


def main() -> None:
    events = parse("examples/sample_dataset.csv")
    baseline = simulate_baseline(events)
    adaptive = simulate_adaptive(events)
    print("Baseline:", baseline)
    print("Adaptive:", adaptive)
    print("Comparison:", compare(baseline, adaptive))


if __name__ == "__main__":
    main()
