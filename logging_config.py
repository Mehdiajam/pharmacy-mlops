# ============================================================
# logging_config.py — Structured Logging Configuration
# ============================================================

import logging
import json
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

# Create logs directory if it doesn't exist
LOGS_DIR = "/app/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# JSON LOGGING SETUP
# ─────────────────────────────────────────

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno


def setup_logging(app_name: str = "mlops_monitoring"):
    """Setup structured logging for the application"""
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # ─────────────────────────────────────────
    # Console Handler (plain text)
    # ─────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # ─────────────────────────────────────────
    # File Handler - JSON logs (all events)
    # ─────────────────────────────────────────
    json_file_path = os.path.join(LOGS_DIR, f"{app_name}.json")
    json_handler = RotatingFileHandler(
        json_file_path,
        maxBytes=10_000_000,  # 10 MB
        backupCount=10
    )
    json_handler.setLevel(logging.DEBUG)
    json_formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    json_handler.setFormatter(json_formatter)
    logger.addHandler(json_handler)
    
    # ─────────────────────────────────────────
    # File Handler - Errors only
    # ─────────────────────────────────────────
    error_file_path = os.path.join(LOGS_DIR, f"{app_name}_errors.log")
    error_handler = RotatingFileHandler(
        error_file_path,
        maxBytes=10_000_000,  # 10 MB
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)
    
    # ─────────────────────────────────────────
    # File Handler - Anomalies & Alerts
    # ─────────────────────────────────────────
    alerts_file_path = os.path.join(LOGS_DIR, f"{app_name}_alerts.log")
    alerts_handler = RotatingFileHandler(
        alerts_file_path,
        maxBytes=10_000_000,  # 10 MB
        backupCount=10
    )
    alerts_handler.setLevel(logging.WARNING)
    alerts_formatter = logging.Formatter(
        fmt='%(asctime)s | ALERT | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    alerts_handler.setFormatter(alerts_formatter)
    logger.addHandler(alerts_handler)
    
    return logger


# ─────────────────────────────────────────
# LOGGING UTILITIES
# ─────────────────────────────────────────

def log_prediction_event(prediction_data: dict):
    """Log prediction event with context"""
    logger = logging.getLogger(__name__)
    logger.info("PREDICTION", extra={
        "event_type": "prediction",
        "confidence": prediction_data.get("confidence"),
        "latency_ms": prediction_data.get("latency_ms"),
        "model_version": prediction_data.get("model_version"),
    })


def log_anomaly(anomaly_type: str, severity: str, details: dict):
    """Log detected anomaly"""
    logger = logging.getLogger(__name__)
    level = logging.ERROR if severity == "critical" else logging.WARNING
    logger.log(level, f"ANOMALY_DETECTED: {anomaly_type}", extra={
        "event_type": "anomaly",
        "anomaly_type": anomaly_type,
        "severity": severity,
        "details": details,
    })


def log_drift_event(drift_type: str, metric_name: str, baseline_value: float, current_value: float):
    """Log drift detection event"""
    logger = logging.getLogger(__name__)
    logger.warning(f"DRIFT_DETECTED: {drift_type}", extra={
        "event_type": "drift_detection",
        "drift_type": drift_type,
        "metric_name": metric_name,
        "baseline_value": baseline_value,
        "current_value": current_value,
        "deviation_percent": f"{((baseline_value - current_value) / baseline_value * 100):.1f}%",
    })


def log_retraining_trigger(reason: str, metrics: dict):
    """Log when retraining is triggered"""
    logger = logging.getLogger(__name__)
    logger.warning(f"RETRAINING_TRIGGERED: {reason}", extra={
        "event_type": "retraining_trigger",
        "reason": reason,
        "triggered_metrics": metrics,
    })


def log_api_error(endpoint: str, error_message: str, error_code: int = 500):
    """Log API errors"""
    logger = logging.getLogger(__name__)
    logger.error(f"API_ERROR: {endpoint}", extra={
        "event_type": "api_error",
        "endpoint": endpoint,
        "error_code": error_code,
        "error_message": error_message,
    })


def log_model_health_check(model_name: str, health_status: dict):
    """Log model health check results"""
    logger = logging.getLogger(__name__)
    logger.info(f"MODEL_HEALTH_CHECK: {model_name}", extra={
        "event_type": "model_health",
        "model_name": model_name,
        "status": health_status.get("status"),
        "accuracy": health_status.get("accuracy"),
        "confidence_mean": health_status.get("confidence_mean"),
    })


# Initialize logging on module import
logger = setup_logging("pharmacy_mlops")
