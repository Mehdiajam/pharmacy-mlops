# ============================================================
# api.py — FastAPI Prediction Endpoint with Prometheus Monitoring
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import requests
import asyncio
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging
import json

# Import monitoring & logging
from monitoring import get_monitoring_aggregator
from logging_config import log_prediction_event, log_api_error, log_anomaly

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
# Model directory can be overridden in container/runtime via MODEL_DIR.
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_DIR = os.path.abspath(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))

print(f"DEBUG: MODEL_DIR set to: {MODEL_DIR}")
print(f"DEBUG: MODEL_DIR exists: {os.path.exists(MODEL_DIR)}")

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
profit_model = None
features = None
encoders = None
sarima_model = None
prophet_model = None
ts_data = None

@app.on_event("startup")
def load_model():
    global model, profit_model, features, encoders, sarima_model, prophet_model, ts_data
    model_path    = f"{MODEL_DIR}/rf_model.pkl"
    profit_model_path = f"{MODEL_DIR}/profit_model.pkl"
    features_path = f"{MODEL_DIR}/features.pkl"
    encoders_path = f"{MODEL_DIR}/encoders.pkl"
    sarima_path   = f"{MODEL_DIR}/sarima_model.pkl"
    prophet_path  = f"{MODEL_DIR}/prophet_model.pkl"
    ts_data_path  = f"{MODEL_DIR}/ts_data.pkl"
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        model    = joblib.load(model_path)
        features = joblib.load(features_path)
        print(f"✅ Classification model loaded from {model_path}")
        
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)
            print(f"✅ Encoders loaded for consistent inference")
        else:
            print(f"⚠️  No encoders found. Categorical predictions might be inaccurate.")
    else:
        print(f"⚠️  No classification model found at {model_path}. Run train.py first.")

    if os.path.exists(profit_model_path):
        profit_model = joblib.load(profit_model_path)
        print(f"✅ Profitability model loaded from {profit_model_path}")
    else:
        print(f"⚠️  No profitability model found at {profit_model_path}.")
    
    # Load forecasting models if available
    if os.path.exists(sarima_path):
        sarima_model = joblib.load(sarima_path)
        print(f"✅ SARIMA model loaded from {sarima_path}")
    else:
        print(f"⚠️  No SARIMA model found. Forecasting will be unavailable.")
    
    if os.path.exists(prophet_path):
        prophet_model = joblib.load(prophet_path)
        print(f"✅ Prophet model loaded from {prophet_path}")
    else:
        print(f"⚠️  No Prophet model found. Forecasting will be unavailable.")
    
    if os.path.exists(ts_data_path):
        ts_data = joblib.load(ts_data_path)
        print(f"✅ Time series data loaded from {ts_data_path}")

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

class PredictProfitResponse(BaseModel):
    predicted_volume: float
    predicted_profit: float
    margin_per_unit:  float
    roi_percent:      float

class ForecastResponse(BaseModel):
    model:             str
    forecast_values:   List[float]
    forecast_dates:    List[str]
    actual_values:     List[float]
    actual_dates:      List[str]
    rmse:              float
    mae:               float

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

    if features is None:
        api_errors_total.labels(endpoint="/predict", error_type="features_not_loaded").inc()
        prediction_requests_total.labels(status="error", model_type="none").inc()
        raise HTTPException(status_code=503, detail="Features not loaded. Run the training pipeline first.")

    try:
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

        # 1. ENCODE CATEGORICAL FEATURES (using saved encoders)
        if encoders is None:
            api_errors_total.labels(endpoint="/predict", error_type="encoders_not_loaded").inc()
            prediction_requests_total.labels(status="error", model_type="random_forest").inc()
            raise HTTPException(status_code=503, detail="Model encoders are not loaded. Re-train the model or verify MODEL_DIR.")

        for col, le in encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].iloc[0])
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    input_df[col] = -1
                    logging.warning(f"Unknown category for {col}: {val}. Setting placeholder -1.")

        # 2. ENSURE ALL FEATURES EXIST (handle missing)
        for feat in features:
            if feat not in input_df.columns:
                input_df[feat] = 0

        # 3. ALIGN WITH TRAINING FEATURE ORDER
        input_df = input_df[features]

        # 4. Verify no object columns remain
        object_cols = input_df.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            logging.error(f"Unencoded object columns remain before prediction: {object_cols}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction cannot proceed: unencoded categorical fields present: {object_cols}"
            )

        input_df = input_df.astype(float)
        # Make prediction
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
        error_msg = f"Prediction error: {str(e)}"
        print(f"ERROR in /predict: {error_msg}")
        import traceback
        print(traceback.format_exc())
        
        api_errors_total.labels(endpoint="/predict", error_type="prediction_error").inc()
        prediction_requests_total.labels(status="error", model_type="random_forest").inc()
        monitoring.record_prediction(latency=latency, confidence=0.0, error=True)
        log_api_error("/predict", str(e), 500)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/predict/profitability", response_model=PredictProfitResponse, summary="Predict product profitability")
def predict_profitability(request: PredictRequest):
    """Predict expected sales volume and profitability"""
    start_time = time.time()
    
    if profit_model is None:
        prediction_requests_total.labels(status="error", model_type="random_forest_regressor").inc()
        raise HTTPException(status_code=503, detail="Profitability model not loaded. Run the training pipeline first.")

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

    # 1. ENCODE CATEGORICAL FEATURES
    if encoders:
        for col, le in encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].iloc[0])
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    input_df[col] = -1 
    
    # 2. ENSURE ALL FEATURES EXIST
    for feat in features:
        if feat not in input_df.columns:
            input_df[feat] = 0

    # 3. ALIGN WITH TRAINING FEATURE ORDER
    input_df = input_df[features]

    try:
        predicted_volume = float(profit_model.predict(input_df)[0])
        predicted_profit = predicted_volume * margin_ht
        roi_percent      = (margin_ht / (request.Prix_d_achat_HT + 1e-9)) * 100

        # Record metrics
        latency = time.time() - start_time
        prediction_latency_seconds.observe(latency)
        prediction_requests_total.labels(status="success", model_type="random_forest_regressor").inc()

        return PredictProfitResponse(
            predicted_volume=round(predicted_volume, 2),
            predicted_profit=round(predicted_profit, 2),
            margin_per_unit=round(margin_ht, 2),
            roi_percent=round(roi_percent, 2)
        )
    except Exception as e:
        prediction_requests_total.labels(status="error", model_type="random_forest_regressor").inc()
        log_api_error("/predict/profitability", str(e), 500)
        raise HTTPException(status_code=500, detail=f"Profit prediction error: {str(e)}")

@app.get("/forecast", summary="Get time series forecast")
def get_forecast(model_type: str = "prophet", periods: int = 6):
    """Get stock quantity forecast using Prophet or SARIMA"""
    
    # Validate models are loaded
    if ts_data is None:
        raise HTTPException(status_code=503, detail="Time series data not available. Run training pipeline first.")
    
    if model_type == "prophet" and prophet_model is None:
        raise HTTPException(status_code=503, detail="Prophet model not loaded. Run training pipeline first.")
    
    if model_type == "sarima" and sarima_model is None:
        raise HTTPException(status_code=503, detail="SARIMA model not loaded. Run training pipeline first.")
    
    try:
        # Validate ts_data structure
        if not isinstance(ts_data, dict) or 'train' not in ts_data or 'test' not in ts_data:
            raise ValueError("Time series data structure is invalid. Expected dict with 'train' and 'test' keys.")
        
        train = ts_data['train']
        test = ts_data['test']
        
        if model_type == "prophet":
            if prophet_model is None:
                raise HTTPException(status_code=503, detail="Prophet model not loaded")
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Get Prophet forecast for test period
            future = prophet_model.make_future_dataframe(periods=len(test), freq='MS')
            forecast_full = prophet_model.predict(future)
            forecast_data = forecast_full.set_index('ds')['yhat'].iloc[-len(test):]
            
            rmse = np.sqrt(mean_squared_error(test.values, forecast_data.values))
            mae = mean_absolute_error(test.values, forecast_data.values)
            
            return ForecastResponse(
                model="Prophet",
                forecast_values=[float(v) for v in forecast_data.values],
                forecast_dates=[d.strftime("%Y-%m-%d") for d in forecast_data.index],
                actual_values=[float(v) for v in test.values],
                actual_dates=[d.strftime("%Y-%m-%d") for d in test.index],
                rmse=round(rmse, 2),
                mae=round(mae, 2)
            )
        
        elif model_type == "sarima":
            if sarima_model is None:
                raise HTTPException(status_code=503, detail="SARIMA model not loaded")
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            forecast_data = sarima_model.forecast(steps=len(test))
            forecast_data.index = test.index
            
            rmse = np.sqrt(mean_squared_error(test.values, forecast_data.values))
            mae = mean_absolute_error(test.values, forecast_data.values)
            
            return ForecastResponse(
                model="SARIMA",
                forecast_values=[float(v) for v in forecast_data.values],
                forecast_dates=[d.strftime("%Y-%m-%d") for d in forecast_data.index],
                actual_values=[float(v) for v in test.values],
                actual_dates=[d.strftime("%Y-%m-%d") for d in test.index],
                rmse=round(rmse, 2),
                mae=round(mae, 2)
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"Forecasting error: {str(e)}. Traceback: {traceback.format_exc()}"
        print(error_msg)
        api_errors_total.labels(endpoint="/forecast", error_type="forecast_error").inc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/forecast/history", summary="Get historical time series data")
def get_forecast_history():
    """Get training history for visualization"""
    
    if ts_data is None:
        raise HTTPException(status_code=503, detail="Time series data not available. Run training pipeline first.")
    
    try:
        # Validate ts_data structure
        if not isinstance(ts_data, dict) or 'train' not in ts_data or 'test' not in ts_data:
            raise ValueError("Time series data structure is invalid. Expected dict with 'train' and 'test' keys.")
        
        train = ts_data['train']
        test = ts_data['test']
        
        return {
            "train": {
                "dates": [d.strftime("%Y-%m-%d") for d in train.index],
                "values": [float(v) for v in train.values]
            },
            "test": {
                "dates": [d.strftime("%Y-%m-%d") for d in test.index],
                "values": [float(v) for v in test.values]
            },
            "total_points": len(train) + len(test),
            "train_points": len(train),
            "test_points": len(test)
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"Error retrieving history: {str(e)}. Traceback: {traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

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