"""ML benchmarking pipeline for task completion prediction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_DEADLINE_BUCKETS = ("none", "<6", "6-24", "24-72", ">72")


@dataclass
class _TaskInstance:
    task_id: str
    created_at: Any
    next_created_at: Any
    category: str
    deadline_bucket: str
    hour_of_day: int
    weekday: int


def _bucket_deadline(deadline_hours: float | None) -> str:
    if deadline_hours is None:
        return "none"
    if deadline_hours < 6:
        return "<6"
    if deadline_hours < 24:
        return "6-24"
    if deadline_hours <= 72:
        return "24-72"
    return ">72"


def _normalize_category(category: str | None) -> str:
    if category is None:
        return "unknown"
    normalized = str(category).strip()
    return normalized or "unknown"


def _build_instances(events: list) -> list[_TaskInstance]:
    events_sorted = sorted(events, key=lambda e: (e.task_id, e.timestamp, e.action))
    by_task: dict[str, list] = defaultdict(list)
    for event in events_sorted:
        by_task[event.task_id].append(event)

    instances: list[_TaskInstance] = []
    for task_id, task_events in by_task.items():
        created_events = [event for event in task_events if event.action == "created"]
        for index, created in enumerate(created_events):
            next_created_at = created_events[index + 1].timestamp if index + 1 < len(created_events) else None
            instances.append(
                _TaskInstance(
                    task_id=f"{task_id}#{index + 1}",
                    created_at=created.timestamp,
                    next_created_at=next_created_at,
                    category=_normalize_category(created.category),
                    deadline_bucket=_bucket_deadline(created.deadline_hours),
                    hour_of_day=int(created.timestamp.hour),
                    weekday=int(created.timestamp.weekday()),
                )
            )

    return sorted(instances, key=lambda inst: (inst.created_at, inst.task_id))


def _label_instance(instance: _TaskInstance, events: list) -> int:
    base_task_id = instance.task_id.split("#", maxsplit=1)[0]
    for event in events:
        if event.task_id != base_task_id:
            continue
        if event.action != "done":
            continue
        if event.timestamp <= instance.created_at:
            continue
        if instance.next_created_at is not None and event.timestamp >= instance.next_created_at:
            continue
        return 1
    return 0


def build_training_table(events: list) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a deterministic task-instance level table (X, y, feature_names)."""

    if not events:
        return np.empty((0, 0)), np.array([], dtype=int), []

    instances = _build_instances(events)
    if not instances:
        return np.empty((0, 0)), np.array([], dtype=int), []

    categories = sorted({instance.category for instance in instances})
    feature_names = ["hour_of_day", "weekday", "rolling_success_rate"]
    feature_names += [f"deadline_bucket={bucket}" for bucket in _DEADLINE_BUCKETS]
    feature_names += [f"category={category}" for category in categories]

    rows: list[list[float]] = []
    labels: list[int] = []
    successes = 0

    events_sorted = sorted(events, key=lambda e: (e.task_id, e.timestamp, e.action))

    for seen, instance in enumerate(instances):
        label = _label_instance(instance, events_sorted)
        rolling_success_rate = (successes / seen) if seen else 0.5

        row = [float(instance.hour_of_day), float(instance.weekday), float(rolling_success_rate)]
        row.extend(1.0 if instance.deadline_bucket == bucket else 0.0 for bucket in _DEADLINE_BUCKETS)
        row.extend(1.0 if instance.category == category else 0.0 for category in categories)

        rows.append(row)
        labels.append(label)
        successes += label

    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), feature_names


def _make_models(seed: int) -> dict[str, Any]:
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
    }


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def benchmark_models(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """Benchmark candidate models with stratified split + CV."""

    if len(X) == 0 or len(y) == 0:
        return {"models": {}, "best_model": None}

    models = _make_models(seed)
    class_counts = np.bincount(y) if len(np.unique(y)) > 1 else np.array([len(y)])
    can_stratify = len(np.unique(y)) > 1 and int(class_counts.min()) >= 2 and len(y) >= 4
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y if can_stratify else None,
    )

    min_class_count = int(np.bincount(y_train).min()) if len(np.unique(y_train)) > 1 else 1
    cv_folds = max(2, min(5, min_class_count)) if len(y_train) >= 2 else 2
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    report: dict[str, Any] = {"models": {}}
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}

    for name, model in models.items():
        model_metrics: dict[str, Any] = {}

        if len(np.unique(y_train)) > 1 and min_class_count >= 2:
            cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
            model_metrics["cv"] = {
                metric: {
                    "mean": float(np.mean(cv_scores[f"test_{metric}"])),
                    "std": float(np.std(cv_scores[f"test_{metric}"])),
                }
                for metric in ("roc_auc", "f1", "accuracy")
            }
        else:
            model_metrics["cv"] = {
                "roc_auc": {"mean": 0.5, "std": 0.0},
                "f1": {"mean": 0.0, "std": 0.0},
                "accuracy": {"mean": float(np.mean(y_train == y_train[0])), "std": 0.0},
            }

        fitted = model.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)
        if hasattr(fitted, "predict_proba"):
            y_score = fitted.predict_proba(X_test)[:, 1]
        else:
            y_score = y_pred.astype(float)

        model_metrics["test"] = {
            "roc_auc": _safe_roc_auc(y_test, y_score),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
        }
        report["models"][name] = model_metrics

    ranked = sorted(
        report["models"].items(),
        key=lambda item: item[1]["cv"]["roc_auc"]["mean"],
        reverse=True,
    )
    report["ranking"] = [
        {"model": name, "cv_roc_auc_mean": metrics["cv"]["roc_auc"]["mean"]} for name, metrics in ranked
    ]
    report["best_model"] = ranked[0][0] if ranked else None
    return report


def train_best_model(X: np.ndarray, y: np.ndarray) -> tuple[Any, dict]:
    """Train the best-performing model (by CV ROC-AUC) on full data."""

    report = benchmark_models(X, y, seed=42)
    best_name = report.get("best_model")
    if best_name is None:
        raise ValueError("Cannot train best model on empty dataset")

    model = _make_models(seed=42)[best_name]
    model.fit(X, y)
    return model, report
