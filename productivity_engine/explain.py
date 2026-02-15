"""Model explainability helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline


def _extract_estimator(model: Any) -> Any:
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def explain_model(model: Any, feature_names: list[str]) -> dict:
    """Return top-10 most important features for linear/tree models."""

    estimator = _extract_estimator(model)

    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).ravel()
        kind = "coefficients"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).ravel()
        kind = "feature_importances"
    else:
        return {"type": "unsupported", "top_features": []}

    pairs = sorted(zip(feature_names, values), key=lambda item: abs(item[1]), reverse=True)[:10]
    return {
        "type": kind,
        "top_features": [{"feature": feature, "weight": float(weight)} for feature, weight in pairs],
    }
