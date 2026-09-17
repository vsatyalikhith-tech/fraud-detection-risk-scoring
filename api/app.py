"""
app.py
--------
FastAPI Risk Scoring API for the Fraud Detection & Risk Scoring project.

Run:
    uvicorn api.app:app --reload --port 8000

Then POST to http://127.0.0.1:8000/predict with a JSON body, e.g.:

{
  "amount": 8500,
  "hour": 2,
  "day_of_week": 5,
  "merchant_category": "electronics",
  "payment_type": "credit_card",
  "device": "web_chrome",
  "new_device": 1,
  "transactions_last_1h": 8,
  "transactions_last_24h": 12,
  "cust_running_avg_amount": 1500
}

Interactive docs: http://127.0.0.1:8000/docs
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from predict import score_transaction  # noqa: E402

app = FastAPI(
    title="Fraud Detection & Risk Scoring API",
    description="Scores a transaction and returns fraud probability, risk score, and risk level.",
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, example=8500)
    hour: int = Field(12, ge=0, le=23, example=2)
    day_of_week: int = Field(0, ge=0, le=6, description="0=Monday ... 6=Sunday")
    merchant_category: Optional[str] = Field(None, example="electronics")
    payment_type: Optional[str] = Field(None, example="credit_card")
    device: Optional[str] = Field(None, example="web_chrome")
    new_device: int = Field(0, ge=0, le=1)
    transactions_last_1h: float = 0
    transactions_last_24h: float = 1
    cust_running_avg_amount: Optional[float] = None
    location_change: float = 0.0


class RiskResponse(BaseModel):
    fraud_probability: float
    risk_score: int
    risk_level: str
    recommended_action: str


@app.get("/")
def root():
    return {"status": "ok", "service": "fraud-detection-risk-scoring-api"}


@app.post("/predict", response_model=RiskResponse)
def predict(txn: TransactionRequest):
    try:
        result = score_transaction(txn.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
