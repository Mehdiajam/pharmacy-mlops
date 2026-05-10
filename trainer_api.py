# trainer_api.py - Complete API with sync and async modes and Prometheus monitoring
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import json
import threading
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ─────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────
training_duration_seconds = Histogram(
    'training_duration_seconds',
    'Training duration in seconds',
    buckets=(30, 60, 120, 300, 600)
)

training_attempts = Counter(
    'training_attempts_total',
    'Total training attempts',
    ['trigger_source', 'status']
)

active_training_jobs = Gauge(
    'active_training_jobs',
    'Number of active training jobs'
)

model_versions_created = Counter(
    'model_versions_created_total',
    'Total model versions created'
)

app = FastAPI(title="Trainer Service API", port=8001)

# Add CORS middleware for n8n integration
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your n8n domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    trigger_source: str = "api"
    force_retrain: bool = False
    async_mode: bool = False  # If True, returns immediately without waiting

class TrainResponse(BaseModel):
    status: str
    message: str
    model_version: Optional[str] = None  # Allow None
    logs: Optional[str] = None           # Allow None
    error: Optional[str] = None          # Allow None

import tempfile

def run_training_background(trigger_source: str):
    """Run training in background thread"""
    lock_file = os.path.join(tempfile.gettempdir(), "training.lock")
    start_time = datetime.now()
    active_training_jobs.inc()
    
    try:
        print(f"[{start_time}] Background training started (trigger: {trigger_source})")
        
        result = subprocess.run(
            ["python", "/app/train.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        training_duration_seconds.observe(duration)
        
        # Remove lock file
        if os.path.exists(lock_file):
            os.remove(lock_file)
        
        if result.returncode == 0:
            print(f"[{datetime.now()}] Training completed successfully")
            training_attempts.labels(trigger_source=trigger_source, status="success").inc()
            model_versions_created.inc()
            # Get model version
            try:
                import mlflow
                from mlflow.tracking import MlflowClient
                mlflow.set_tracking_uri("http://host.docker.internal:5000")
                client = MlflowClient()
                latest_versions = client.get_latest_versions("stock-classifier", stages=["None"])
                if latest_versions:
                    print(f"[{datetime.now()}] New model version: {latest_versions[0].version}")
            except:
                pass
        else:
            print(f"[{datetime.now()}] Training failed: {result.stderr[:200]}")
            training_attempts.labels(trigger_source=trigger_source, status="failed").inc()
            
    except subprocess.TimeoutExpired:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        print(f"[{datetime.now()}] Training timeout after 10 minutes")
        training_attempts.labels(trigger_source=trigger_source, status="timeout").inc()
    except Exception as e:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        print(f"[{datetime.now()}] Training error: {e}")
        training_attempts.labels(trigger_source=trigger_source, status="error").inc()
    finally:
        active_training_jobs.dec()

@app.post("/train", response_model=TrainResponse)
async def trigger_training(request: TrainRequest):
    """Trigger the training pipeline"""
    
    # Check if training is already running
    lock_file = os.path.join(tempfile.gettempdir(), "training.lock")
    if os.path.exists(lock_file):
        return TrainResponse(
            status="busy",
            message="Training already in progress",
            error="Previous training still running"
        )
    
    # Create lock file
    with open(lock_file, "w") as f:
        f.write(str(os.getpid()))
    
    # ASYNC MODE: Return immediately, run in background
    if request.async_mode:
        print(f"[{datetime.now()}] Async training triggered by: {request.trigger_source}")
        
        # Start training in background thread
        thread = threading.Thread(target=run_training_background, args=(request.trigger_source,))
        thread.daemon = True
        thread.start()
        
        return TrainResponse(
            status="started",
            message="Training started in background. Check MLflow UI at http://localhost:5000 for results",
            model_version=None,
            logs=None,
            error=None
        )
    
    # SYNC MODE: Wait for training to complete
    try:
        print(f"[{datetime.now()}] Sync training triggered by: {request.trigger_source}")
        
        start_time = datetime.now()
        active_training_jobs.inc()
        
        result = subprocess.run(
            ["python", "/app/train.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        training_duration_seconds.observe(duration)
        active_training_jobs.dec()
        
        # Remove lock file
        os.remove(lock_file)
        
        if result.returncode == 0:
            training_attempts.labels(trigger_source=request.trigger_source, status="success").inc()
            model_versions_created.inc()
            # Get the latest model version from MLflow
            model_version = "unknown"
            try:
                import mlflow
                from mlflow.tracking import MlflowClient
                mlflow.set_tracking_uri("http://host.docker.internal:5000")
                client = MlflowClient()
                latest_versions = client.get_latest_versions("stock-classifier", stages=["None"])
                if latest_versions:
                    model_version = latest_versions[0].version
            except:
                pass
            
            return TrainResponse(
                status="success",
                message=f"Training completed successfully (trigger: {request.trigger_source})",
                model_version=model_version,
                logs=result.stdout[-2000:] if result.stdout else "No output",
                error=None
            )
        else:
            training_attempts.labels(trigger_source=request.trigger_source, status="failed").inc()
            return TrainResponse(
                status="failed",
                message="Training failed",
                model_version=None,
                logs=result.stdout[-1000:] if result.stdout else "No output",
                error=result.stderr[-1000:] if result.stderr else "Unknown error"
            )
            
    except subprocess.TimeoutExpired:
        active_training_jobs.dec()
        if os.path.exists(lock_file):
            os.remove(lock_file)
        training_attempts.labels(trigger_source=request.trigger_source, status="timeout").inc()
        return TrainResponse(
            status="failed",
            message="Training timeout after 10 minutes",
            model_version=None,
            logs=None,
            error="Process took too long"
        )
    except Exception as e:
        active_training_jobs.dec()
        if os.path.exists(lock_file):
            os.remove(lock_file)
        training_attempts.labels(trigger_source=request.trigger_source, status="error").inc()
        return TrainResponse(
            status="error",
            message="Unexpected error",
            model_version=None,
            logs=None,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trainer"}

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/train/profitability", response_model=TrainResponse)
async def trigger_profitability_training(request: TrainRequest):
    """Trigger only the profitability regressor training"""
    
    # Check if training is already running
    lock_file = os.path.join(tempfile.gettempdir(), "profit_training.lock")
    if os.path.exists(lock_file):
        return TrainResponse(
            status="busy",
            message="Profitability training already in progress",
            error="Previous profitability training still running"
        )
    
    # Create lock file
    with open(lock_file, "w") as f:
        f.write(str(os.getpid()))
    
    # ASYNC MODE: Return immediately, run in background
    if request.async_mode:
        print(f"[{datetime.now()}] Async profitability training triggered by: {request.trigger_source}")
        
        # Start training in background thread
        def run_profitability_training_background(trigger_source: str):
            lock_file = os.path.join(tempfile.gettempdir(), "profit_training.lock")
            start_time = datetime.now()
            active_training_jobs.inc()
            
            try:
                print(f"[{start_time}] Background profitability training started (trigger: {trigger_source})")
                
                # Run only profitability training script
                result = subprocess.run(
                    ["python", "/app/train.py"],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes timeout for profitability only
                    env={**os.environ, "TRAIN_PROFITABILITY_ONLY": "true"}
                )
                
                duration = (datetime.now() - start_time).total_seconds()
                training_duration_seconds.observe(duration)
                
                # Remove lock file
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                
                if result.returncode == 0:
                    print(f"[{datetime.now()}] Profitability training completed successfully")
                    training_attempts.labels(trigger_source=trigger_source, status="success").inc()
                    model_versions_created.inc()
                else:
                    print(f"[{datetime.now()}] Profitability training failed: {result.stderr[:200]}")
                    training_attempts.labels(trigger_source=trigger_source, status="failed").inc()
                    
            except subprocess.TimeoutExpired:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                print(f"[{datetime.now()}] Profitability training timeout after 5 minutes")
                training_attempts.labels(trigger_source=trigger_source, status="timeout").inc()
            except Exception as e:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                print(f"[{datetime.now()}] Profitability training error: {e}")
                training_attempts.labels(trigger_source=trigger_source, status="error").inc()
            finally:
                active_training_jobs.dec()
        
        thread = threading.Thread(target=run_profitability_training_background, args=(request.trigger_source,))
        thread.daemon = True
        thread.start()
        
        return TrainResponse(
            status="started",
            message="Profitability training started in background. Check MLflow UI for results",
            model_version=None,
            logs=None,
            error=None
        )
    
    # SYNC MODE: Wait for training to complete
    try:
        print(f"[{datetime.now()}] Sync profitability training triggered by: {request.trigger_source}")
        
        start_time = datetime.now()
        active_training_jobs.inc()
        
        # Run only profitability training
        result = subprocess.run(
            ["python", "/app/train.py"],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "TRAIN_PROFITABILITY_ONLY": "true"}
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        training_duration_seconds.observe(duration)
        
        if result.returncode == 0:
            training_attempts.labels(trigger_source=request.trigger_source, status="success").inc()
            model_versions_created.inc()
            return TrainResponse(
                status="success",
                message="Profitability training completed successfully",
                model_version="profitability-regressor",
                logs=result.stdout[-500:],  # Last 500 chars of logs
                error=None
            )
        else:
            training_attempts.labels(trigger_source=request.trigger_source, status="failed").inc()
            return TrainResponse(
                status="failed",
                message="Profitability training failed",
                model_version=None,
                logs=result.stdout[-500:],
                error=result.stderr[-500:]
            )
            
    except subprocess.TimeoutExpired:
        training_attempts.labels(trigger_source=request.trigger_source, status="timeout").inc()
        return TrainResponse(
            status="timeout",
            message="Profitability training timed out after 5 minutes",
            model_version=None,
            logs=None,
            error="Training timeout"
        )
    except Exception as e:
        training_attempts.labels(trigger_source=request.trigger_source, status="error").inc()
        return TrainResponse(
            status="error",
            message=f"Profitability training error: {str(e)}",
            model_version=None,
            logs=None,
            error=str(e)
        )
    finally:
        active_training_jobs.dec()
        # Clean up lock file
        lock_file = os.path.join(tempfile.gettempdir(), "profit_training.lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)