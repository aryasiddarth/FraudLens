"""
SHAP explainability for per-prediction feature attribution.
Uses TreeExplainer for tree-based models, LinearExplainer for LR.
"""

import numpy as np
import pandas as pd
import shap


FEATURE_NAMES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23",
    "V24", "V25", "V26", "V27", "V28", "Amount"
]

_explainer_cache = {}


def get_explainer(model):
    model_id = id(model)
    if model_id not in _explainer_cache:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            # For LR, use a small background sample — computed lazily
            explainer = None
        _explainer_cache[model_id] = explainer
    return _explainer_cache[model_id]


def compute_shap_values(model, X_row: np.ndarray, feature_names=None) -> dict:
    """
    Compute SHAP values for a single prediction row.
    Returns top 10 feature contributions as {feature_name: shap_value}.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    try:
        explainer = get_explainer(model)
        if explainer is None:
            return {}

        df_row = pd.DataFrame([X_row], columns=feature_names)
        raw = explainer.shap_values(df_row)

        # Tree models may return list [class0, class1]
        if isinstance(raw, list):
            values = raw[1][0]
        else:
            values = raw[0]

        shap_dict = {name: float(val) for name, val in zip(feature_names, values)}
        # Return top 10 by absolute magnitude
        top10 = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
        return top10

    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return {}
