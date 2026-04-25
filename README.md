# 📦 Pharmacy Stock MLOps Pipeline

A complete MLOps system for predicting pharmacy inventory stock levels — built with MLflow, FastAPI, Streamlit, and n8n, fully containerized with Docker.

\---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Docker Network                       │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Streamlit│───▶│ FastAPI  │───▶│  Trainer API     │  │
│  │  :8501   │    │  :8000   │    │    :8001         │  │
│  └──────────┘    └────┬─────┘    └────────┬─────────┘  │
│                       │                   │             │
│                  ┌────▼─────┐        ┌────▼─────┐      │
│                  │  MLflow  │        │ train.py │      │
│                  │  :5000   │        │ (ML pipe)│      │
│                  └──────────┘        └──────────┘      │
│                                                         │
│  ┌──────────┐                                           │
│  │   n8n    │─── Orchestrates the entire pipeline       │
│  │  :5678   │                                           │
│  └──────────┘                                           │
└─────────────────────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  PostgreSQL (host)  │
              │  parapharmacie\_BD   │
              └─────────────────────┘
```

\---

## 📁 Project Structure

```
n8n\_work/
├── docker-compose.yaml     # Orchestrates all services
├── Dockerfile.api          # FastAPI container
├── Dockerfile.train        # Trainer container
├── requirements.txt        # Python dependencies
│
├── train.py                # ML training pipeline (MLflow tracked)
├── trainer\_api.py          # Trainer FastAPI service
├── api.py                  # Prediction FastAPI service
├── app.py                  # Streamlit web application
│
├── models/                 # Saved model artifacts
│   ├── rf\_model.pkl
│   └── features.pkl
│
├── mlflow\_data/            # MLflow tracking database \& artifacts
├── n8n\_data/               # n8n workflows \& credentials
└── mlproject/              # ML project files (mounted volume)
```

\---

## 🚀 Getting Started

### Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
* PostgreSQL running locally with database `parapharmacie\_BD`
* At least **10GB** of free disk space
* At least **8GB** of RAM

### 1\. Clone the repository

```bash
git clone https://github.com/YOUR\_USERNAME/pharmacy-mlops.git
cd pharmacy-mlops
```

### 2\. Create required folders

```bash
mkdir models
mkdir mlflow\_data
```

### 3\. Start all services

```bash
docker compose up -d --build
```

> ⚠️ First build takes 10-15 minutes — it compiles scikit-learn, xgboost, prophet, etc.

### 4\. Verify all containers are running

```bash
docker ps
```

You should see:

|Container|Port|Status|
|-|-|-|
|`mlflow`|5000|✅ Up|
|`fastapi`|8000|✅ Up|
|`trainer`|8001|✅ Up|
|`streamlit\_app`|8501|✅ Up|
|`n8n\_school\_project`|5678|✅ Up|

\---

## 🧪 Running the ML Pipeline

### Trigger training via API

```bash
curl -X POST http://localhost:8000/train \\
  -H "Content-Type: application/json" \\
  -d '{"trigger\_source": "manual", "async\_mode": true}'
```

### Monitor training in MLflow

Open **http://localhost:5000** → you will see:

* Experiment: `stock-classification`
* Run 1: `logistic-regression`
* Run 2: `random-forest`

Each run tracks:

* **Parameters**: model hyperparameters (C, penalty, n\_estimators, max\_depth...)
* **Metrics**: accuracy, precision, recall, F1-score, ROC-AUC
* **Artifacts**: confusion matrix, ROC curve, feature importance plots

\---

## 🔮 Making Predictions

### Via FastAPI (Swagger UI)

Open **http://localhost:8000/docs** → click `/predict` → Try it out

### Via curl

```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "Prix\_d\_achat\_HT": 25.5,
    "Prix\_de\_vente\_HT": 35.0,
    "Prix\_de\_vente\_TTC": 41.3,
    "Famille": "TOILETTE",
    "Designation": "SHAMPOO",
    "Libelle": "SHAMPOO 200ML",
    "Ville": "Tunis"
  }'
```

**Response:**

```json
{
  "prediction": 1,
  "label": "Low Stock",
  "confidence": 0.8611,
  "probability\_low": 0.8611,
  "probability\_high": 0.1389
}
```

### Via Streamlit Web App

Open **http://localhost:8501** — fill in the form and click **Predict Stock Level**

\---

## 🤖 ML Pipeline Details

### Dataset

* **211 records** from a pharmacy inventory database (PostgreSQL)
* **Star schema**: FactInventory + Dim\_Produit + DimDate + DimLocalisation
* **8 product families**: TOILETTE, COMPRIME, COMP. ALIM., OPH-ORL, POMMADE, SUPPOSITOIRE, DIVERS, NATURE

### Target Variable

`Stock\_Level` = 1 (Low Stock) if `Quantite ≤ 10`, else 0 (High Stock)

### Models Trained

|Model|Accuracy|F1-Score|ROC-AUC|
|-|-|-|-|
|Logistic Regression|0.6977|0.7937|0.7103|
|**Random Forest** ✅|**0.7674**|**0.8611**|**0.8056**|

### Pipeline Steps

1. **Load** — fetch data from PostgreSQL via SQLAlchemy
2. **Preprocess** — encode categoricals, handle missing values
3. **Balance** — apply SMOTE to handle class imbalance
4. **Train** — GridSearchCV with StratifiedKFold (5 splits)
5. **Evaluate** — compute metrics, generate plots
6. **Log** — track everything in MLflow
7. **Save** — persist model to `/models/rf\_model.pkl`

\---

## 🔄 n8n Workflow — Automated Pipeline

Import `mlops\_workflow.json` into n8n (**http://localhost:5678**):

1. Open n8n → **Add workflow** → **Import from file**
2. Select `mlops\_workflow.json`
3. Click **Execute workflow** to run manually

### Workflow Steps

```
Daily Schedule (8AM)
       ↓
Trigger Training (POST /train)
       ↓
Wait 15 minutes
       ↓
Health Check (GET /health)
       ↓
Model Ready? ──NO──▶ Log Error
       │
      YES
       ↓
Run Prediction (POST /predict)
       ↓
Low Stock? ──NO──▶ Log "Stock OK ✅"
       │
      YES
       ↓
Alert "Low Stock 🔴"
```

\---

## 🌐 Services \& URLs

|Service|URL|Description|
|-|-|-|
|MLflow UI|http://localhost:5000|Experiment tracking \& model registry|
|FastAPI Docs|http://localhost:8000/docs|Interactive API documentation|
|Streamlit App|http://localhost:8501|Web UI for predictions|
|n8n|http://localhost:5678|Workflow automation|
|Trainer API|http://localhost:8001/docs|Training trigger service|

\---

## 📡 API Endpoints

### Prediction API (port 8000)

|Method|Endpoint|Description|
|-|-|-|
|GET|`/`|Root — API status|
|GET|`/health`|Health check + model status|
|POST|`/predict`|Predict stock level|
|POST|`/train`|Trigger training pipeline|
|GET|`/model/info`|Model metadata \& features|

### Trainer API (port 8001)

|Method|Endpoint|Description|
|-|-|-|
|POST|`/train`|Run training (sync or async)|
|GET|`/health`|Trainer health check|

\---

## 🛠️ Useful Commands

```bash
# View logs of a specific service
docker logs fastapi --tail 50
docker logs trainer -f
docker logs mlflow --tail 20

# Restart a single service
docker compose restart fastapi

# Stop everything
docker compose down

# Rebuild a specific service
docker compose up -d --build fastapi

# Remove training lock if stuck
docker exec trainer rm -f /tmp/training.lock

# Check disk usage
docker system df
```

\---

## ✅ Assignment Checklist (S12 MLOps)

|Requirement|Status|
|-|-|
|Experiment tracking (MLflow) — 2+ runs|✅|
|Automated training pipeline|✅|
|Model versioning (MLflow registry)|✅|
|Model serving (FastAPI `/predict`)|✅|
|Containerization (Docker Compose)|✅|
|Clean, structured code|✅|
|Web App → API integration (Streamlit)|✅|
|Workflow automation (n8n)|✅|

\---

## 🗃️ Tech Stack

|Layer|Technology|
|-|-|
|ML Framework|scikit-learn, XGBoost, Prophet|
|Experiment Tracking|MLflow 2.13.0|
|API|FastAPI + Uvicorn|
|Web UI|Streamlit|
|Workflow Automation|n8n|
|Database|PostgreSQL|
|Containerization|Docker + Docker Compose|
|Language|Python 3.11|

\---

## 👤 Author

**Mehdi Ajam** — Pharmacy Inventory ML Project  
School project — S12 MLOps Assignment

