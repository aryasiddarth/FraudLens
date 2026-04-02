"""
Model and artifact loading utilities for FraudLens backend.
Loads best_model.pkl, scaler.pkl, and threshold.json at startup.
"""

import os
import json
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


class ModelRegistry:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = 0.5
        self.model_name = "Unknown"
        self.metrics = {}
        self.feature_names = []
        self.scaled_columns = []
        self.target_column = None
        self.dataset_path = None
        self.raw_features = []
        self.numeric_raw_features = []
        self.categorical_raw_features = []
        self.numeric_medians = {}

    def load(self):
        model_path = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        threshold_path = os.path.join(ARTIFACTS_DIR, "threshold.json")
        metadata_path = os.path.join(ARTIFACTS_DIR, "metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run `python ml/train.py` first."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                info = json.load(f)
            self.threshold = info.get("threshold", 0.5)
            self.model_name = info.get("model_name", "Unknown")
            self.metrics = info.get("metrics", {})

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
            self.scaled_columns = meta.get("scaled_columns", [])
            self.target_column = meta.get("target_column")
            self.dataset_path = meta.get("dataset_path")
            self.raw_features = meta.get("raw_features", [])
            self.numeric_raw_features = meta.get("numeric_raw_features", [])
            self.categorical_raw_features = meta.get("categorical_raw_features", [])
            self.numeric_medians = meta.get("numeric_medians", {})

        print(f"✅  Loaded model: {self.model_name}")
        print(f"    Threshold: {self.threshold:.4f}")


# Global singleton
registry = ModelRegistry()
