from datetime import datetime

from productivity_engine.ml_pipeline import benchmark_models, build_training_table
from productivity_engine.schema import NormalizedEvent


def sample_events():
    return [
        NormalizedEvent("t1", datetime.fromisoformat("2025-01-01T09:00:00"), "created", 30, "work"),
        NormalizedEvent("t1", datetime.fromisoformat("2025-01-01T11:00:00"), "done", 28, "work"),
        NormalizedEvent("t2", datetime.fromisoformat("2025-01-02T10:00:00"), "created", 10, "health"),
        NormalizedEvent("t2", datetime.fromisoformat("2025-01-02T13:00:00"), "ignore", 7, "health"),
        NormalizedEvent("t3", datetime.fromisoformat("2025-01-03T08:00:00"), "created", None, None),
        NormalizedEvent("t3", datetime.fromisoformat("2025-01-03T20:00:00"), "done", 68, "study"),
    ]


def test_build_training_table_smoke():
    X, y, feature_names = build_training_table(sample_events())
    assert X.shape[0] == len(y)
    assert X.shape[1] == len(feature_names)
    assert "hour_of_day" in feature_names
    assert "weekday" in feature_names
    assert "rolling_success_rate" in feature_names


def test_benchmark_models_smoke():
    X, y, _ = build_training_table(sample_events())
    report = benchmark_models(X, y, seed=42)

    assert "models" in report
    assert "best_model" in report
    assert report["best_model"] in report["models"]

    for name in ("LogisticRegression", "RandomForest", "GradientBoosting"):
        assert name in report["models"]
        roc_auc = report["models"][name]["test"]["roc_auc"]
        assert 0.0 <= roc_auc <= 1.0
