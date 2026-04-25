# ============================================================
# api.py — FastAPI Prediction Endpoint
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import requests
import asyncio

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_DIR   = os.getenv("MODEL_DIR",  "/app/models")
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
TRAINER_URL = os.getenv("TRAINER_API_URL", "http://trainer:8001")
MODEL_NAME  = "stock-classifier"

mlflow.set_tracking_uri(MLFLOW_URI)

app = FastAPI(
    title="Stock Classification API",
    description="Predicts whether a pharmacy product is Low Stock or High Stock",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────
model    = None
features = None

@app.on_event("startup")
def load_model():
    global model, features
    model_path    = f"{MODEL_DIR}/rf_model.pkl"
    features_path = f"{MODEL_DIR}/features.pkl"
    if os.path.exists(model_path) and os.path.exists(features_path):
        model    = joblib.load(model_path)
        features = joblib.load(features_path)
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"⚠️  No model found at {model_path}. Run train.py first.")

# ─────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────
class TrainRequest(BaseModel):
    trigger_source: str  = "api"
    force_retrain:  bool = False
    async_mode:     bool = True  # Default to async so it returns immediately

class TrainResponse(BaseModel):
    status:        str
    message:       str
    model_version: Optional[str] = None
    logs:          Optional[str] = None
    error:         Optional[str] = None

class PredictRequest(BaseModel):
    Prix_d_achat_HT:     float          = Field(..., example=25.5)
    Prix_de_vente_HT:    float          = Field(..., example=35.0)
    Prix_de_vente_TTC:   float          = Field(..., example=41.3)
    Famille:             str            = Field(..., example="TOILETTE")
    Designation:         str            = Field(..., example="SHAMPOO")
    Libelle:             str            = Field(..., example="SHAMPOO 200ML")
    Ville:               str            = Field(..., example="Tunis")
    Margin_HT:           Optional[float] = None
    Margin_Ratio:        Optional[float] = None
    Stock_Duration_Days: Optional[float] = None

class PredictResponse(BaseModel):
    prediction:       int
    label:            str
    confidence:       float
    probability_low:  float
    probability_high: float

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_path:   str

# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────
@app.get("/", summary="Root")
def root():
    return {"message": "Stock Classification API is running", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    return HealthResponse(
        status="ok" if model is not None else "no_model",
        model_loaded=model is not None,
        model_path=f"{MODEL_DIR}/rf_model.pkl"
    )

@app.post("/train", response_model=TrainResponse, summary="Trigger model training")
async def trigger_training(request: TrainRequest):
    """Forward training request to the trainer service"""

    def make_request():
        response = requests.post(
            f"{TRAINER_URL}/train",
            json={
                "trigger_source": request.trigger_source,
                "force_retrain":  request.force_retrain,
                "async_mode":     request.async_mode   # ← pass async_mode!
            },
            timeout=30  # Short timeout — trainer handles the long work
        )
        response.raise_for_status()
        return response.json()

    try:
        result = await asyncio.to_thread(make_request)
        return TrainResponse(**result)
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Trainer service timed out")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot reach trainer service")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse, summary="Predict stock level")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training pipeline first.")

    margin_ht    = request.Margin_HT    if request.Margin_HT    is not None else request.Prix_de_vente_HT - request.Prix_d_achat_HT
    margin_ratio = request.Margin_Ratio if request.Margin_Ratio is not None else margin_ht / (request.Prix_d_achat_HT + 1e-9)

    raw_input = {
        "Prix_d_achat_HT":    request.Prix_d_achat_HT,
        "Prix_de_vente_HT":   request.Prix_de_vente_HT,
        "Prix_de_vente_TTC":  request.Prix_de_vente_TTC,
        "Famille":             request.Famille,
        "Designation":         request.Designation,
        "Libelle":             request.Libelle,
        "Ville":               request.Ville,
        "Margin_HT":           margin_ht,
        "Margin_Ratio":        margin_ratio,
        "Stock_Duration_Days": request.Stock_Duration_Days if request.Stock_Duration_Days is not None else 0.0,
    }

    input_df = pd.DataFrame([raw_input])

    for feat in features:
        if feat not in input_df.columns:
            input_df[feat] = 0

    for col in input_df.select_dtypes(include='object').columns:
        if col in features:
            input_df[col] = input_df[col].astype('category').cat.codes

    input_df = input_df[features]

    try:
        prediction    = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        prob_high     = float(probabilities[0])
        prob_low      = float(probabilities[1])
        confidence    = max(prob_high, prob_low)
        label         = "Low Stock" if prediction == 1 else "High Stock"

        return PredictResponse(
            prediction=prediction,
            label=label,
            confidence=round(confidence, 4),
            probability_low=round(prob_low, 4),
            probability_high=round(prob_high, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info", summary="Model information")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return {
        "model_type": type(model.best_estimator_).__name__ if hasattr(model, 'best_estimator_') else type(model).__name__,
        "n_features": len(features),
        "features":   features,
        "model_path": f"{MODEL_DIR}/rf_model.pkl"
    }