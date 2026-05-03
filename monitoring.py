# ============================================================
# monitoring.py — Drift Detection & Model Health Monitoring
# ============================================================

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# BASELINE DEFINITIONS
# ─────────────────────────────────────────
BASELINE_FILE = "/app/baseline_metrics.json"

DEFAULT_BASELINE = {
    "model_accuracy": 0.85,  # Will be updated after first training
    "model_confidence_mean": 0.75,
    "model_confidence_std": 0.15,
    "latency_p95": 0.5,  # seconds
    "latency_p99": 1.0,  # seconds
    "error_rate": 0.01,  # 1%
    "data_missing_ratio": 0.0,
    "timestamp": datetime.now().isoformat()
}

# ─────────────────────────────────────────
# THRESHOLDS FOR ANOMALIES
# ─────────────────────────────────────────
ACCURACY_DROP_THRESHOLD = 0.05  # 5% drop
CONFIDENCE_DROP_THRESHOLD = 0.10  # 10% drop
ERROR_RATE_THRESHOLD = 0.05  # 5% error rate
LATENCY_THRESHOLD_P95 = 2.0  # 2 seconds
LATENCY_THRESHOLD_P99 = 5.0  # 5 seconds
DRIFT_KS_THRESHOLD = 0.15  # Kolmogorov-Smirnov test threshold


class BaselineManager:
    """Manages baseline metrics and loads/saves from file"""
    
    def __init__(self, baseline_file: str = BASELINE_FILE):
        self.baseline_file = baseline_file
        self.baseline = self._load_baseline()
    
    def _load_baseline(self) -> Dict:
        """Load baseline from file or use defaults"""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, 'r') as f:
                    baseline = json.load(f)
                logger.info(f"✅ Baseline loaded from {self.baseline_file}")
                return baseline
            except Exception as e:
                logger.warning(f"⚠️  Failed to load baseline: {e}. Using defaults.")
                return DEFAULT_BASELINE.copy()
        else:
            logger.info(f"📊 No baseline found. Creating new baseline at {self.baseline_file}")
            return DEFAULT_BASELINE.copy()
    
    def save_baseline(self):
        """Save baseline to file"""
        try:
            self.baseline["timestamp"] = datetime.now().isoformat()
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baseline, f, indent=2)
            logger.info(f"✅ Baseline saved to {self.baseline_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save baseline: {e}")
    
    def update_baseline(self, metrics: Dict):
        """Update baseline with new metrics"""
        self.baseline.update(metrics)
        self.save_baseline()
    
    def get_baseline(self) -> Dict:
        """Get current baseline"""
        return self.baseline.copy()


class DriftDetector:
    """Detects data drift and model degradation"""
    
    def __init__(self, baseline_manager: BaselineManager):
        self.baseline_manager = baseline_manager
        self.baseline = baseline_manager.get_baseline()
    
    def detect_accuracy_degradation(self, current_accuracy: float) -> Tuple[bool, str]:
        """Check if accuracy dropped below threshold"""
        baseline_acc = self.baseline.get("model_accuracy", 0.85)
        drop = baseline_acc - current_accuracy
        drop_percentage = (drop / baseline_acc) * 100
        
        if drop > ACCURACY_DROP_THRESHOLD:
            return True, f"🚨 Accuracy degradation: {baseline_acc:.4f} → {current_accuracy:.4f} (-{drop_percentage:.1f}%)"
        return False, f"✅ Accuracy stable: {current_accuracy:.4f}"
    
    def detect_confidence_shift(self, current_confidence_mean: float, 
                               current_confidence_std: float) -> Tuple[bool, str]:
        """Check if confidence distribution shifted significantly"""
        baseline_conf = self.baseline.get("model_confidence_mean", 0.75)
        conf_drop = baseline_conf - current_confidence_mean
        
        if conf_drop > CONFIDENCE_DROP_THRESHOLD:
            return True, f"⚠️  Confidence shift: {baseline_conf:.4f} → {current_confidence_mean:.4f} (std: {current_confidence_std:.4f})"
        return False, f"✅ Confidence stable: {current_confidence_mean:.4f} (std: {current_confidence_std:.4f})"
    
    def detect_data_drift_ks(self, baseline_distribution: Optional[List[float]], 
                             current_distribution: List[float]) -> Tuple[bool, str, float]:
        """
        Kolmogorov-Smirnov test for data drift detection.
        Compares current data distribution with baseline.
        """
        if baseline_distribution is None or len(current_distribution) < 10:
            return False, "📊 Insufficient data for drift detection", 0.0
        
        try:
            statistic, p_value = stats.ks_2samp(baseline_distribution, current_distribution)
            
            if statistic > DRIFT_KS_THRESHOLD:
                return True, f"🚨 Data drift detected (KS stat: {statistic:.4f}, p-value: {p_value:.4f})", statistic
            
            return False, f"✅ No data drift (KS stat: {statistic:.4f}, p-value: {p_value:.4f})", statistic
        except Exception as e:
            logger.error(f"❌ KS test error: {e}")
            return False, f"Error in drift detection: {str(e)}", 0.0
    
    def detect_error_rate_spike(self, current_error_rate: float) -> Tuple[bool, str]:
        """Check if error rate exceeded threshold"""
        baseline_error = self.baseline.get("error_rate", 0.01)
        
        if current_error_rate > ERROR_RATE_THRESHOLD:
            return True, f"🚨 Error rate spike: {current_error_rate:.2%} (baseline: {baseline_error:.2%})"
        
        return False, f"✅ Error rate normal: {current_error_rate:.2%}"
    
    def detect_latency_degradation(self, p95: float, p99: float) -> Tuple[bool, str]:
        """Check if latency exceeded thresholds"""
        baseline_p95 = self.baseline.get("latency_p95", 0.5)
        baseline_p99 = self.baseline.get("latency_p99", 1.0)
        
        alerts = []
        if p95 > LATENCY_THRESHOLD_P95:
            alerts.append(f"P95 latency: {p95:.3f}s (threshold: {LATENCY_THRESHOLD_P95}s)")
        if p99 > LATENCY_THRESHOLD_P99:
            alerts.append(f"P99 latency: {p99:.3f}s (threshold: {LATENCY_THRESHOLD_P99}s)")
        
        if alerts:
            return True, f"⚠️  Latency degradation: {', '.join(alerts)}"
        
        return False, f"✅ Latency normal: P95={p95:.3f}s, P99={p99:.3f}s"
    
    def detect_data_freshness_issue(self, last_data_timestamp: datetime) -> Tuple[bool, str]:
        """Check if data is stale"""
        time_since_update = datetime.now() - last_data_timestamp
        stale_threshold = 86400  # 24 hours
        
        if time_since_update.total_seconds() > stale_threshold:
            return True, f"⚠️  Data freshness issue: Last update {time_since_update.days} days ago"
        
        return False, f"✅ Data fresh: Updated {time_since_update.total_seconds():.0f}s ago"


class MonitoringAggregator:
    """Aggregates metrics and generates monitoring reports"""
    
    def __init__(self):
        self.baseline_manager = BaselineManager()
        self.drift_detector = DriftDetector(self.baseline_manager)
        self.recent_metrics = {
            "predictions": [],
            "latencies": [],
            "confidences": [],
            "errors": 0,
            "total_requests": 0,
        }
    
    def record_prediction(self, latency: float, confidence: float, error: bool = False):
        """Record a prediction for monitoring"""
        self.recent_metrics["predictions"].append({
            "timestamp": datetime.now().isoformat(),
            "latency": latency,
            "confidence": confidence
        })
        self.recent_metrics["latencies"].append(latency)
        self.recent_metrics["confidences"].append(confidence)
        self.recent_metrics["total_requests"] += 1
        
        if error:
            self.recent_metrics["errors"] += 1
        
        # Keep only last 1000 predictions in memory
        if len(self.recent_metrics["predictions"]) > 1000:
            self.recent_metrics["predictions"] = self.recent_metrics["predictions"][-1000:]
            self.recent_metrics["latencies"] = self.recent_metrics["latencies"][-1000:]
            self.recent_metrics["confidences"] = self.recent_metrics["confidences"][-1000:]
    
    def generate_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        if not self.recent_metrics["total_requests"]:
            return {"status": "no_data", "message": "No predictions recorded yet"}
        
        latencies = self.recent_metrics["latencies"]
        confidences = self.recent_metrics["confidences"]
        
        # Calculate metrics
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        mean_latency = np.mean(latencies)
        
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        error_rate = self.recent_metrics["errors"] / self.recent_metrics["total_requests"]
        
        # Run drift detection
        acc_drift, acc_msg = self.drift_detector.detect_accuracy_degradation(mean_confidence)
        conf_drift, conf_msg = self.drift_detector.detect_confidence_shift(mean_confidence, std_confidence)
        error_spike, error_msg = self.drift_detector.detect_error_rate_spike(error_rate)
        latency_issue, latency_msg = self.drift_detector.detect_latency_degradation(p95_latency, p99_latency)
        
        is_healthy = not (acc_drift or conf_drift or error_spike or latency_issue)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy" if is_healthy else "degraded",
            "metrics": {
                "total_requests": self.recent_metrics["total_requests"],
                "error_count": self.recent_metrics["errors"],
                "error_rate": f"{error_rate:.2%}",
                "latency_p50_ms": f"{p50_latency*1000:.1f}",
                "latency_p95_ms": f"{p95_latency*1000:.1f}",
                "latency_p99_ms": f"{p99_latency*1000:.1f}",
                "latency_mean_ms": f"{mean_latency*1000:.1f}",
                "confidence_mean": f"{mean_confidence:.4f}",
                "confidence_std": f"{std_confidence:.4f}",
            },
            "alerts": [
                {"type": "accuracy_degradation", "triggered": acc_drift, "message": acc_msg},
                {"type": "confidence_shift", "triggered": conf_drift, "message": conf_msg},
                {"type": "error_rate_spike", "triggered": error_spike, "message": error_msg},
                {"type": "latency_degradation", "triggered": latency_issue, "message": latency_msg},
            ],
            "triggered_alerts": [a for a in [acc_msg, conf_msg, error_msg, latency_msg] if "🚨" in a or "⚠️" in a],
        }
        
        return report
    
    def reset_metrics(self):
        """Reset metrics for new monitoring period"""
        self.recent_metrics = {
            "predictions": [],
            "latencies": [],
            "confidences": [],
            "errors": 0,
            "total_requests": 0,
        }


# Global instance
_monitoring_aggregator = None

def get_monitoring_aggregator() -> MonitoringAggregator:
    """Get or create global monitoring aggregator"""
    global _monitoring_aggregator
    if _monitoring_aggregator is None:
        _monitoring_aggregator = MonitoringAggregator()
    return _monitoring_aggregator
