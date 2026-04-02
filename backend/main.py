"""
FraudLens FastAPI Backend
Endpoints:
  GET  /                  Health check
  GET  /model/info        Model name, metrics, threshold
  POST /predict           Single transaction fraud prediction + SHAP
  POST /predict/batch     Batch prediction
  GET  /plots/{filename}  Serve training visualization PNGs
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(ROOT, "artifacts", "plots")

from backend.schemas import (
    TransactionInput, PredictionResponse,
    ModelInfoResponse, BatchTransactionInput, BatchPredictionResponse,
)
from backend.model_loader import registry
from backend.shap_explainer import compute_shap_values


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts once at startup."""
    registry.load()
    yield


app = FastAPI(
    title="FraudLens API",
    description="Real-time loan default risk detection with ML + SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount plots directory for static image serving
os.makedirs(PLOTS_DIR, exist_ok=True)
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")


# ─── Helpers ───────────────────────────────────────────────────────────────────
def ensure_registry_metadata():
    if registry.feature_names:
        return
    metadata_path = os.path.join(ROOT, "artifacts", "metadata.json")
    if os.path.exists(metadata_path):
        # Artifacts may have been retrained after API startup; refresh once.
        registry.load()
    if not registry.feature_names:
        raise HTTPException(status_code=503, detail="Model metadata missing. Re-run training.")


def preprocess_transaction(tx: TransactionInput) -> np.ndarray:
    ensure_registry_metadata()

    raw = {
        "NAME_CONTRACT_TYPE": tx.NAME_CONTRACT_TYPE,
        "CODE_GENDER": tx.CODE_GENDER,
        "FLAG_OWN_CAR": tx.FLAG_OWN_CAR,
        "FLAG_OWN_REALTY": tx.FLAG_OWN_REALTY,
        "NAME_INCOME_TYPE": tx.NAME_INCOME_TYPE,
        "NAME_EDUCATION_TYPE": tx.NAME_EDUCATION_TYPE,
        "CNT_CHILDREN": tx.CNT_CHILDREN,
        "AMT_INCOME_TOTAL": tx.AMT_INCOME_TOTAL,
        "AMT_CREDIT": tx.AMT_CREDIT,
        "AMT_ANNUITY": tx.AMT_ANNUITY,
        "AMT_GOODS_PRICE": tx.AMT_GOODS_PRICE,
        "DAYS_BIRTH": tx.DAYS_BIRTH,
        "DAYS_EMPLOYED": tx.DAYS_EMPLOYED,
        "EXT_SOURCE_1": tx.EXT_SOURCE_1,
        "EXT_SOURCE_2": tx.EXT_SOURCE_2,
        "EXT_SOURCE_3": tx.EXT_SOURCE_3,
    }
    row = pd.DataFrame([raw])

    numeric_cols = registry.numeric_raw_features or []
    categorical_cols = registry.categorical_raw_features or []
    medians = registry.numeric_medians or {}

    for col in numeric_cols:
        row[col] = row[col].fillna(medians.get(col, 0.0))
    for col in categorical_cols:
        row[col] = row[col].fillna("MISSING").astype(str)

    row = pd.get_dummies(row, columns=categorical_cols, drop_first=True)
    row = row.reindex(columns=registry.feature_names, fill_value=0.0)

    scaled_cols = [c for c in (registry.scaled_columns or []) if c in row.columns]
    if scaled_cols:
        row[scaled_cols] = registry.scaler.transform(row[scaled_cols])

    return row.iloc[0].to_numpy(dtype=float)


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    elif prob < 0.85:
        return "HIGH"
    else:
        return "CRITICAL"


def make_prediction(tx: TransactionInput, include_shap: bool = True) -> PredictionResponse:
    processed = preprocess_transaction(tx)

    prob = float(registry.model.predict_proba([processed])[0, 1])
    is_fraud = prob >= registry.threshold
    risk = get_risk_level(prob)

    shap_vals = {}
    top_features = []
    if include_shap:
        shap_feature_names = registry.feature_names
        shap_vals = compute_shap_values(registry.model, processed, shap_feature_names)
        top_features = [
            {"feature": k, "value": round(v, 6), "direction": "fraud" if v > 0 else "legit"}
            for k, v in sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
        ]

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 6),
        risk_level=risk,
        threshold_used=round(registry.threshold, 6),
        shap_values=shap_vals if shap_vals else None,
        top_features=top_features if top_features else None,
    )


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", summary="Health Check")
def root():
    return {
        "status": "ok",
        "service": "FraudLens API",
        "model": registry.model_name,
    }


@app.get("/model/info", response_model=ModelInfoResponse, summary="Model Info")
def model_info():
    return ModelInfoResponse(
        model_name=registry.model_name,
        threshold=round(registry.threshold, 6),
        metrics=registry.metrics,
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict Fraud")
def predict(transaction: TransactionInput):
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    return make_prediction(transaction, include_shap=True)


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Batch Predict")
def predict_batch(body: BatchTransactionInput):
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(body.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1000.")

    predictions = [make_prediction(tx, include_shap=False) for tx in body.transactions]
    fraud_count = sum(1 for p in predictions if p.is_fraud)

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=fraud_count,
    )
