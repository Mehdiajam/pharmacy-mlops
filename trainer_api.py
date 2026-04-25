# trainer_api.py - Complete API with sync and async modes
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import json
import threading
from datetime import datetime

app = FastAPI(title="Trainer Service API", port=8001)

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

def run_training_background(trigger_source: str):
    """Run training in background thread"""
    lock_file = "/tmp/training.lock"
    try:
        print(f"[{datetime.now()}] Background training started (trigger: {trigger_source})")
        
        result = subprocess.run(
            ["python", "/app/train.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        # Remove lock file
        if os.path.exists(lock_file):
            os.remove(lock_file)
        
        if result.returncode == 0:
            print(f"[{datetime.now()}] Training completed successfully")
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
            
    except subprocess.TimeoutExpired:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        print(f"[{datetime.now()}] Training timeout after 10 minutes")
    except Exception as e:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        print(f"[{datetime.now()}] Training error: {e}")

@app.post("/train", response_model=TrainResponse)
async def trigger_training(request: TrainRequest):
    """Trigger the training pipeline"""
    
    # Check if training is already running
    lock_file = "/tmp/training.lock"
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
        
        result = subprocess.run(
            ["python", "/app/train.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        # Remove lock file
        os.remove(lock_file)
        
        if result.returncode == 0:
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
            return TrainResponse(
                status="failed",
                message="Training failed",
                model_version=None,
                logs=result.stdout[-1000:] if result.stdout else "No output",
                error=result.stderr[-1000:] if result.stderr else "Unknown error"
            )
            
    except subprocess.TimeoutExpired:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        return TrainResponse(
            status="failed",
            message="Training timeout after 10 minutes",
            model_version=None,
            logs=None,
            error="Process took too long"
        )
    except Exception as e:
        if os.path.exists(lock_file):
            os.remove(lock_file)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)