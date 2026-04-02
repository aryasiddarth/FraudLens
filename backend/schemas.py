from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class TransactionInput(BaseModel):
    NAME_CONTRACT_TYPE: str = Field(..., description="Loan contract type (e.g., Cash loans, Revolving loans)")
    CODE_GENDER: str = Field(..., description="Client gender code")
    FLAG_OWN_CAR: str = Field(..., description="Y/N indicator if client owns a car")
    FLAG_OWN_REALTY: str = Field(..., description="Y/N indicator if client owns realty")
    NAME_INCOME_TYPE: str = Field(..., description="Income source category")
    NAME_EDUCATION_TYPE: str = Field(..., description="Education level category")
    CNT_CHILDREN: float = Field(..., ge=0, description="Number of children")
    AMT_INCOME_TOTAL: float = Field(..., ge=0, description="Total annual income")
    AMT_CREDIT: float = Field(..., ge=0, description="Requested credit amount")
    AMT_ANNUITY: float = Field(..., ge=0, description="Loan annuity")
    AMT_GOODS_PRICE: float = Field(..., ge=0, description="Goods price for financed purchase")
    DAYS_BIRTH: float = Field(..., description="Client age in days (negative in source dataset)")
    DAYS_EMPLOYED: float = Field(..., description="Days employed (negative in source dataset)")
    EXT_SOURCE_1: float = Field(..., ge=0, le=1, description="External risk score 1")
    EXT_SOURCE_2: float = Field(..., ge=0, le=1, description="External risk score 2")
    EXT_SOURCE_3: float = Field(..., ge=0, le=1, description="External risk score 3")

    model_config = {
        "json_schema_extra": {
            "example": {
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "F",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Higher education",
                "CNT_CHILDREN": 0,
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 450000,
                "AMT_ANNUITY": 25000,
                "AMT_GOODS_PRICE": 405000,
                "DAYS_BIRTH": -14000,
                "DAYS_EMPLOYED": -2000,
                "EXT_SOURCE_1": 0.45,
                "EXT_SOURCE_2": 0.62,
                "EXT_SOURCE_3": 0.51
            }
        }
    }


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    threshold_used: float
    shap_values: Optional[Dict[str, float]] = None
    top_features: Optional[list] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    threshold: float
    metrics: Dict[str, Any]


class BatchTransactionInput(BaseModel):
    transactions: list[TransactionInput]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    fraud_count: int
