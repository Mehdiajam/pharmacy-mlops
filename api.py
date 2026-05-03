# ============================================================
# api.py — FastAPI Prediction Endpoint with Prometheus Monitoring
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import requests
import asyncio
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging

# Import monitoring & logging
from monitoring import get_monitoring_aggregator
from logging_config import log_prediction_event, log_api_error, log_anomaly

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_DIR   = os.getenv("MODEL_DIR",  "/app/models")
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
TRAINER_URL = os.getenv("TRAINER_API_URL", "http://trainer:8001")
MODEL_NAME  = "stock-classifier"

mlflow.set_tracking_uri(MLFLOW_URI)

# ─────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────
prediction_requests_total = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['status', 'model_type']
)

prediction_latency_seconds = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0)
)

prediction_confidence = Histogram(
    'prediction_confidence',
    'Prediction confidence distribution',
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
)

model_accuracy_gauge = Gauge(
    'model_accuracy_percent',
    'Current model accuracy percentage'
)

model_confidence_mean = Gauge(
    'model_confidence_mean',
    'Mean prediction confidence'
)

api_errors_total = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type']
)

training_requests_total = Counter(
    'training_requests_total',
    'Total training requests',
    ['status']
)

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

@app.get("/metrics", summary="Prometheus metrics")
def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    return HealthResponse(
        status="ok" if model is not None else "no_model",
        model_loaded=model is not None,
        model_path=f"{MODEL_DIR}/rf_model.pkl"
    )

@app.post("/train", response_model=TrainResponse, summary="Trigger model training")
async def trigger_training(request: TrainRequest):
    """Forward training request to the trainer service with monitoring"""

    def make_request():
        response = requests.post(
            f"{TRAINER_URL}/train",
            json={
                "trigger_source": request.trigger_source,
                "force_retrain":  request.force_retrain,
                "async_mode":     request.async_mode
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    try:
        result = await asyncio.to_thread(make_request)
        training_requests_total.labels(status=result.get("status", "unknown")).inc()
        return TrainResponse(**result)
    except requests.exceptions.Timeout:
        training_requests_total.labels(status="timeout").inc()
        api_errors_total.labels(endpoint="/train", error_type="timeout").inc()
        raise HTTPException(status_code=504, detail="Trainer service timed out")
    except requests.exceptions.ConnectionError:
        training_requests_total.labels(status="connection_error").inc()
        api_errors_total.labels(endpoint="/train", error_type="connection_error").inc()
        raise HTTPException(status_code=503, detail="Cannot reach trainer service")
    except Exception as e:
        training_requests_total.labels(status="error").inc()
        api_errors_total.labels(endpoint="/train", error_type="unknown").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse, summary="Predict stock level")
def predict(request: PredictRequest):
    """Predict stock level with monitoring"""
    start_time = time.time()
    monitoring = get_monitoring_aggregator()
    
    if model is None:
        api_errors_total.labels(endpoint="/predict", error_type="model_not_loaded").inc()
        prediction_requests_total.labels(status="error", model_type="none").inc()
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

        # Record metrics
        latency = time.time() - start_time
        prediction_latency_seconds.observe(latency)
        prediction_confidence.observe(confidence)
        prediction_requests_total.labels(status="success", model_type="random_forest").inc()
        
        # Track in monitoring aggregator
        monitoring.record_prediction(latency=latency, confidence=confidence, error=False)
        
        # Log prediction event
        log_prediction_event({
            "confidence": confidence,
            "latency_ms": latency * 1000,
            "prediction": label,
        })
        
        # Update gauges
        health_report = monitoring.generate_health_report()
        if health_report.get("status") != "no_data":
            try:
                model_confidence_mean.set(float(health_report["metrics"]["confidence_mean"]))
            except:
                pass

        return PredictResponse(
            prediction=prediction,
            label=label,
            confidence=round(confidence, 4),
            probability_low=round(prob_low, 4),
            probability_high=round(prob_high, 4)
        )
    except Exception as e:
        latency = time.time() - start_time
        api_errors_total.labels(endpoint="/predict", error_type="prediction_error").inc()
        prediction_requests_total.labels(status="error", model_type="random_forest").inc()
        monitoring.record_prediction(latency=latency, confidence=0.0, error=True)
        log_api_error("/predict", str(e), 500)
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

@app.get("/health/detailed", summary="Detailed health report with monitoring")
def detailed_health():
    """Get detailed health report including monitoring metrics"""
    monitoring = get_monitoring_aggregator()
    report = monitoring.generate_health_report()
    return report