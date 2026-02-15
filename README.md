# productivity-engine

A deterministic Python library for behavioral compliance modeling on productivity events.

## What it does
- Ingests CSV/JSON task event logs.
- Normalizes to a shared `NormalizedEvent` schema.
- Extracts behavioral features.
- Computes hybrid risk (heuristic + logistic regression).
- Selects adaptive reminder times and deadline escalation schedules.
- Simulates baseline vs adaptive outcomes and compares metric deltas.

## Installation
```bash
pip install -e .
pip install -e .[dev]
```

## Example
```bash
python examples/demo.py
```

## Architecture
```text
adapters -> schema -> features -> risk_model -> scheduling/escalation
                                 \-> simulation -> evaluator
                                 \-> metrics
```

## Evaluation
- Baseline uses observed historical metrics.
- Adaptive simulation applies hybrid risk-derived improvement factors.
- Evaluator reports:
  - completion improvement %
  - delay reduction %
  - ignore reduction %

## ML Benchmarking
- `productivity_engine.ml_pipeline` builds a deterministic task-instance training table from event streams.
- Benchmarked models:
  - LogisticRegression (with StandardScaler)
  - RandomForestClassifier
  - GradientBoostingClassifier
- Metrics reported per model:
  - ROC-AUC
  - F1 score
  - Accuracy
- Evaluation method:
  - Stratified train/test split
  - 5-fold (or smallest valid) stratified cross-validation
- Explainability:
  - Linear models use sorted coefficient magnitudes
  - Tree models use sorted `feature_importances_`

Run benchmarking from the command line:
```bash
python scripts/run_benchmark.py --data examples/sample_dataset.csv
```

The script prints results and saves them to `outputs/benchmark_report.json`.

## Streamlit demo
A presentable Streamlit UI is available in `ui_demo_streamlit/`.

See setup and run instructions here: [`ui_demo_streamlit/README.md`](ui_demo_streamlit/README.md).
