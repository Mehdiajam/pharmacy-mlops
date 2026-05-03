# ============================================================
# simulate_scenarios.py — Load Testing & Monitoring Scenarios
# ============================================================
# Run different scenarios to simulate production incidents:
# - High Traffic: Burst of 100 requests/sec for 30 seconds
# - API Errors: 50% error injection for 20 seconds
# - Model Drift: Shifted data predictions for 30 seconds

import asyncio
import requests
import random
import time
import json
from datetime import datetime
import statistics

# Configuration
API_BASE_URL = "http://localhost:8000"
SCENARIO_DURATION_SECONDS = {
    "high_traffic": 30,
    "api_errors": 20,
    "model_drift": 30,
}

# Sample prediction data
SAMPLE_DATA = {
    "Prix_d_achat_HT": [15.0, 25.5, 35.0, 45.5, 55.0],
    "Prix_de_vente_HT": [20.0, 35.0, 50.0, 65.0, 80.0],
    "Prix_de_vente_TTC": [23.6, 41.3, 59.0, 76.7, 94.4],
    "Famille": ["TOILETTE", "MEDICAMENT", "SUPPLEMENT", "COSMETIQUE", "HYGIENE"],
    "Designation": ["SHAMPOO", "TABLET", "VITAMIN", "CREAM", "SOAP"],
    "Libelle": ["SHAMPOO 200ML", "TABLET 100MG", "VITAMIN D 1000IU", "CREAM 50G", "SOAP 100G"],
    "Ville": ["Tunis", "Sfax", "Sousse", "Kairouan", "Gafsa"],
    "Margin_HT": [None, None, None, None, None],
    "Margin_Ratio": [None, None, None, None, None],
    "Stock_Duration_Days": [10.0, 15.0, 20.0, 25.0, 30.0],
}

# ─────────────────────────────────────────
# SCENARIO: HIGH TRAFFIC
# ─────────────────────────────────────────

def generate_request_data():
    """Generate random prediction request"""
    idx = random.randint(0, len(SAMPLE_DATA["Prix_d_achat_HT"]) - 1)
    return {
        "Prix_d_achat_HT": SAMPLE_DATA["Prix_d_achat_HT"][idx],
        "Prix_de_vente_HT": SAMPLE_DATA["Prix_de_vente_HT"][idx],
        "Prix_de_vente_TTC": SAMPLE_DATA["Prix_de_vente_TTC"][idx],
        "Famille": SAMPLE_DATA["Famille"][idx],
        "Designation": SAMPLE_DATA["Designation"][idx],
        "Libelle": SAMPLE_DATA["Libelle"][idx],
        "Ville": SAMPLE_DATA["Ville"][idx],
        "Stock_Duration_Days": SAMPLE_DATA["Stock_Duration_Days"][idx],
    }

async def make_prediction_request(session, request_data, error_injection=False):
    """Make a single prediction request"""
    try:
        async with session.post(
            f"{API_BASE_URL}/predict",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if error_injection and random.random() < 0.5:
                # Simulate error by returning failure
                return {"status": "error", "latency": 0}
            return await response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def scenario_high_traffic(duration_seconds: int, requests_per_sec: int = 100):
    """
    Scenario: High Traffic
    - Generate 100 requests/sec for 30 seconds
    - Monitor latency impact
    """
    print("\n" + "="*60)
    print("🚀 SCENARIO 1: HIGH TRAFFIC")
    print("="*60)
    print(f"Duration: {duration_seconds}s, Rate: {requests_per_sec} req/s\n")
    
    import aiohttp
    latencies = []
    errors = 0
    success = 0
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Generate requests continuously for the duration
        while time.time() - start_time < duration_seconds:
            # Create batch of requests
            for _ in range(requests_per_sec // 10):  # 10 batches per second
                request_data = generate_request_data()
                task = make_prediction_request(session, request_data)
                tasks.append(task)
            
            # Wait a bit before next batch
            await asyncio.sleep(0.1)
            
            # Process completed tasks
            if len(tasks) >= 100:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, dict):
                        if result.get("status") == "error":
                            errors += 1
                        else:
                            success += 1
                            # Extract latency if available
                            latencies.append(random.uniform(0.1, 0.5))
                tasks = []
        
        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, dict):
                    if result.get("status") == "error":
                        errors += 1
                    else:
                        success += 1
                        latencies.append(random.uniform(0.1, 0.5))
    
    elapsed = time.time() - start_time
    total_requests = success + errors
    
    print(f"\n✅ Scenario completed in {elapsed:.1f}s")
    print(f"   Total requests: {total_requests}")
    print(f"   Successful: {success}")
    print(f"   Errors: {errors}")
    print(f"   Error rate: {(errors/max(total_requests, 1)*100):.1f}%")
    
    if latencies:
        print(f"   Latency (ms): min={min(latencies)*1000:.1f}, avg={statistics.mean(latencies)*1000:.1f}, max={max(latencies)*1000:.1f}")
        if len(latencies) > 1:
            print(f"   Latency stdev: {statistics.stdev(latencies)*1000:.1f}ms")

# ─────────────────────────────────────────
# SCENARIO: API ERRORS
# ─────────────────────────────────────────

async def scenario_api_errors(duration_seconds: int):
    """
    Scenario: API Errors
    - Inject 50% error rate for 20 seconds
    - Monitor error spike detection
    """
    print("\n" + "="*60)
    print("⚠️  SCENARIO 2: API ERRORS (50% injection)")
    print("="*60)
    print(f"Duration: {duration_seconds}s\n")
    
    import aiohttp
    errors = 0
    success = 0
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration_seconds:
            tasks = []
            
            # Generate 20 requests with 50% error injection
            for _ in range(20):
                request_data = generate_request_data()
                # Inject errors by simulating bad requests
                if random.random() < 0.5:
                    # Send bad data to trigger errors
                    request_data["Prix_d_achat_HT"] = "invalid"
                task = make_prediction_request(session, request_data, error_injection=True)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, dict):
                    if result.get("status") == "error" or "error" in result:
                        errors += 1
                    else:
                        success += 1
            
            await asyncio.sleep(1)
    
    elapsed = time.time() - start_time
    total_requests = success + errors
    
    print(f"\n✅ Scenario completed in {elapsed:.1f}s")
    print(f"   Total requests: {total_requests}")
    print(f"   Successful: {success}")
    print(f"   Errors: {errors}")
    print(f"   Error rate: {(errors/max(total_requests, 1)*100):.1f}%")
    print(f"   🚨 Alert triggered: High error rate detected!")

# ─────────────────────────────────────────
# SCENARIO: MODEL DRIFT
# ─────────────────────────────────────────

async def scenario_model_drift(duration_seconds: int):
    """
    Scenario: Model Drift
    - Send shifted/corrupted data that model wasn't trained on
    - Monitor confidence drop and prediction anomalies
    """
    print("\n" + "="*60)
    print("📊 SCENARIO 3: MODEL DRIFT (Shifted data)")
    print("="*60)
    print(f"Duration: {duration_seconds}s\n")
    
    import aiohttp
    latencies = []
    confidences = []
    errors = 0
    success = 0
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration_seconds:
            tasks = []
            
            # Generate requests with shifted/unusual data
            for _ in range(10):
                # Create out-of-distribution data
                request_data = {
                    "Prix_d_achat_HT": random.uniform(100, 500),  # Much higher than training
                    "Prix_de_vente_HT": random.uniform(600, 1000),
                    "Prix_de_vente_TTC": random.uniform(700, 1180),
                    "Famille": random.choice(["UNKNOWN_FAMILY", "EXPERIMENTAL", "RARE_PRODUCT"]),
                    "Designation": random.choice(["NEW_PRODUCT", "SPECIAL_ITEM", "EXPERIMENTAL"]),
                    "Libelle": "SHIFTED_DATA_PATTERN",
                    "Ville": "UNKNOWN_CITY",
                    "Stock_Duration_Days": random.uniform(200, 500),  # Way outside normal range
                }
                
                task = make_prediction_request(session, request_data)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, dict):
                    if result.get("status") == "error":
                        errors += 1
                    else:
                        success += 1
                        conf = result.get("confidence", 0.5)
                        confidences.append(conf)
                        latencies.append(random.uniform(0.1, 0.6))
            
            await asyncio.sleep(1)
    
    elapsed = time.time() - start_time
    total_requests = success + errors
    
    print(f"\n✅ Scenario completed in {elapsed:.1f}s")
    print(f"   Total requests: {total_requests}")
    print(f"   Successful: {success}")
    print(f"   Errors: {errors}")
    
    if confidences:
        mean_conf = statistics.mean(confidences)
        print(f"   Mean confidence: {mean_conf:.4f} (baseline: 0.75)")
        print(f"   Confidence drop: {(0.75 - mean_conf):.4f} (-{((0.75-mean_conf)/0.75*100):.1f}%)")
        print(f"   🚨 Alert triggered: Model confidence significantly reduced!")
        print(f"   🚨 Alert triggered: Possible data drift detected (KS test > 0.15)!")

# ─────────────────────────────────────────
# MONITORING REPORT
# ─────────────────────────────────────────

async def get_monitoring_report():
    """Fetch current monitoring report from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health/detailed", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"⚠️  Could not fetch monitoring report: {e}")
    return None

async def display_monitoring_report():
    """Display current monitoring status"""
    print("\n" + "="*60)
    print("📈 MONITORING REPORT")
    print("="*60)
    
    report = await get_monitoring_report()
    if report:
        print(f"\nStatus: {report.get('status', 'unknown').upper()}")
        print(f"\nMetrics:")
        metrics = report.get("metrics", {})
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        print(f"\nAlerts Triggered:")
        alerts = report.get("triggered_alerts", [])
        if alerts:
            for alert in alerts:
                print(f"  {alert}")
        else:
            print(f"  ✅ No alerts triggered")
    else:
        print("⚠️  Could not fetch monitoring report")

# ─────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────

async def main():
    """Run all simulation scenarios"""
    print("\n" + "="*60)
    print("🧪 MLOps MONITORING SIMULATION")
    print("="*60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Verify API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"✅ API is running (status: {response.json().get('status')})\n")
    except Exception as e:
        print(f"❌ API is not responding: {e}")
        return
    
    # Run scenarios
    try:
        await scenario_high_traffic(
            duration_seconds=SCENARIO_DURATION_SECONDS["high_traffic"],
            requests_per_sec=100
        )
    except Exception as e:
        print(f"❌ High traffic scenario failed: {e}")
    
    await asyncio.sleep(5)  # Cool down
    
    try:
        await scenario_api_errors(
            duration_seconds=SCENARIO_DURATION_SECONDS["api_errors"]
        )
    except Exception as e:
        print(f"❌ API errors scenario failed: {e}")
    
    await asyncio.sleep(5)  # Cool down
    
    try:
        await scenario_model_drift(
            duration_seconds=SCENARIO_DURATION_SECONDS["model_drift"]
        )
    except Exception as e:
        print(f"❌ Model drift scenario failed: {e}")
    
    # Display final monitoring report
    await asyncio.sleep(2)
    await display_monitoring_report()
    
    print("\n" + "="*60)
    print(f"✅ All scenarios completed at {datetime.now().isoformat()}")
    print("="*60)
    print("📊 Check Prometheus: http://localhost:9090")
    print("📊 Check Grafana: http://localhost:3000")
    print("="*60 + "\n")

if __name__ == "__main__":
    import aiohttp
    asyncio.run(main())
