"""Run ML benchmark from a CSV/JSON event dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from productivity_engine.adapters import csv_adapter, json_adapter
from productivity_engine.ml_pipeline import benchmark_models, build_training_table


def _load_events(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return csv_adapter.parse(str(path))
    if suffix == ".json":
        return json_adapter.parse(str(path))
    raise ValueError("Unsupported input format, expected .csv or .json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run productivity-engine ML benchmark")
    parser.add_argument("--data", required=True, help="Path to CSV/JSON events file")
    args = parser.parse_args()

    data_path = Path(args.data)
    events = _load_events(data_path)
    X, y, feature_names = build_training_table(events)
    report = benchmark_models(X, y, seed=42)
    report["feature_names"] = feature_names
    report["n_instances"] = int(len(y))

    print(json.dumps(report, indent=2))

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "benchmark_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved benchmark report to {out_path}")


if __name__ == "__main__":
    main()
