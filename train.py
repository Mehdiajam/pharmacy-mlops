# ============================================================
# train.py — MLflow-tracked Training Pipeline
# Classification: Low/High Stock — Random Forest
# ============================================================

import warnings
warnings.filterwarnings('ignore')
import time
import requests
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import joblib

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DB_URL       = os.getenv("DB_URL", "postgresql://postgres:@host.docker.internal:5432/parapharmacie_BD")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
EXPERIMENT   = "stock-classification"
MODEL_NAME   = "stock-classifier"
MODEL_DIR    = "/app/models"

os.makedirs(MODEL_DIR, exist_ok=True)

def connect_to_mlflow(uri, experiment_name, max_retries=12, retry_interval=5):
    """Waits for MLflow to be ready before proceeding."""
    print(f"Connecting to MLflow at {uri}...")
    for i in range(max_retries):
        try:
            # Check if the health endpoint or base URI is reachable
            requests.get(uri, timeout=2)
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            print("✅ Successfully connected to MLflow and set experiment.")
            return True
        except (requests.exceptions.ConnectionError, Exception) as e:
            print(f"⚠️ MLflow not ready (Attempt {i+1}/{max_retries}). Retrying in {retry_interval}s...")
            time.sleep(retry_interval)
    
    print("❌ Could not connect to MLflow. Check if the container is running.")
    return False

# Trigger the connection before starting the pipeline
if not connect_to_mlflow(MLFLOW_URI, EXPERIMENT):
    exit(1) # Exit with error if connection fails after all retries

# ─────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────
def load_data(db_url: str) -> pd.DataFrame:
    print("[1/6] Loading data from PostgreSQL...")
    engine = create_engine(db_url)
    fact     = pd.read_sql('SELECT * FROM "FactInventory"', engine)
    product  = pd.read_sql('SELECT * FROM "Dim_Produit"', engine)
    date     = pd.read_sql('SELECT * FROM "DimDate"', engine)
    location = pd.read_sql('SELECT * FROM "DimLocalisation"', engine)

    df = fact.merge(product,  left_on="FK_produit",     right_on="produit_id",    how="left").drop(columns=["produit_id"])
    df = df.merge(date,       left_on="FK_Date",         right_on="Date_PK",       how="left").drop(columns=["Date_PK"])
    df = df.merge(location,   left_on="FK_Localisation", right_on="PK_Localisation", how="left").drop(columns=["PK_Localisation"])

    df.rename(columns={'ï»¿Code': 'Code_Localisation'}, inplace=True)
    if 'Quantite_y'         in df.columns: df = df.drop(columns=['Quantite_y'])
    if 'Quantite_d_Alerte_y' in df.columns: df = df.drop(columns=['Quantite_d_Alerte_y'])
    df.rename(columns={'Quantite_x': 'Quantite', 'Quantite_d_Alerte_x': 'Quantite_d_Alerte'}, inplace=True)
    if 'Code postal' in df.columns: df.rename(columns={'Code postal': 'Code_postal'}, inplace=True)

    df['Date_In']  = pd.to_datetime(df['Date_In'],  dayfirst=True, errors='coerce')
    df['Date_Out'] = pd.to_datetime(df['Date_Out'], dayfirst=True, errors='coerce')
    df['Margin_HT']    = df['Prix_de_vente_HT'] - df['Prix_d_achat_HT']
    df['Margin_Ratio'] = df['Margin_HT'] / (df['Prix_d_achat_HT'] + 1e-9)
    df['Stock_Level']  = (df['Quantite'] <= 10).astype(int)

    print(f"    Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df

# ─────────────────────────────────────────
# 2. PREPROCESSING (Fixed - handles all datetime types)
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print("[2/6] Preprocessing...")
    print(f"    Initial columns: {df.columns.tolist()}")
    print(f"    Initial dtypes: {df.dtypes.value_counts().to_dict()}")
    
    # List of columns to explicitly drop
    drop_cols = [
        'PK_Inventory', 'FK_produit', 'FK_Localisation', 'FK_Date',
        'Date_In', 'Date_Out', 'Date', 'Date_code',
        'Quantite', 'Quantite_d_Alerte', 'Stock_Level',
        'Code', 'Code_Localisation', 'Adresse', 'Nom',
        'created_at'
    ]
    
    print(f"    Dropping columns: {[c for c in drop_cols if c in df.columns]}")
    
    # 1. Filter out known drop columns
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols).copy()
    y = df['Stock_Level'].copy()
    
    print(f"    After dropping: X shape {X.shape}, columns: {X.columns.tolist()}")
    
    # 2. AUTOMATIC DATETIME REMOVAL
    datetime_features = X.select_dtypes(include=['datetime64', 'timedelta64', 'datetimetz']).columns.tolist()
    print(f"    Datetime columns found: {datetime_features}")
    
    if datetime_features:
        print(f"    Removing datetime features: {datetime_features}")
        X = X.drop(columns=datetime_features)
    
    # 3. Try to convert object columns to numeric
    print(f"    Object columns before conversion: {X.select_dtypes(include=['object']).columns.tolist()}")
    for col in X.select_dtypes(include=['object']).columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='raise')
            print(f"    Converted {col} to numeric")
        except (ValueError, TypeError) as e:
            print(f"    Could not convert {col}: {str(e)[:50]}...")
            pass
    
    # 4. Encoding categorical variables
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    print(f"    Categorical columns to encode: {cat_cols}")
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"    Encoded {col} with {len(le.classes_)} unique values")
    
    # Final validation
    non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric:
        print(f"    ERROR: Non-numeric columns remaining: {non_numeric}")
        print(f"    Their dtypes: {X[non_numeric].dtypes.to_dict()}")
        raise ValueError(f"Non-numeric columns: {non_numeric}")
    
    features = X.columns.tolist()
    print(f"    ✅ Preprocessing complete: {len(features)} features")
    print(f"    Target distribution: {y.value_counts().to_dict()}")
    return X, y, features, encoders

# ─────────────────────────────────────────
# 3. TRAIN — LOGISTIC REGRESSION (run 1)
# ─────────────────────────────────────────
def train_logistic_regression(X_train, X_test, y_train, y_test, cv):
    print("[3/6] Training Logistic Regression (Run 1)...")
    with mlflow.start_run(run_name="logistic-regression"):
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag("phase", "classification")

        pipeline = ImbPipeline([
            ('smote',  SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('clf',    LogisticRegression(max_iter=1000, random_state=42))
        ])
        params = {
            'clf__C':       [0.01, 0.1, 1, 10],
            'clf__penalty': ['l1', 'l2'],
            'clf__solver':  ['liblinear', 'saga']
        }
        grid = GridSearchCV(pipeline, params, cv=cv, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        y_prob = grid.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "f1_score":  round(f1_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        }
        best_params = {
            "C":       grid.best_params_['clf__C'],
            "penalty": grid.best_params_['clf__penalty'],
            "solver":  grid.best_params_['clf__solver'],
        }

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        # Confusion matrix artifact
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                               display_labels=['High Stock', 'Low Stock']).plot(ax=ax, colorbar=False)
        ax.set_title("LR — Confusion Matrix")
        plt.tight_layout()
        plt.savefig("/tmp/lr_cm.png", dpi=120)
        plt.close()
        mlflow.log_artifact("/tmp/lr_cm.png", artifact_path="plots")

        # Log model
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(grid, "model", signature=signature)

        print(f"    LR metrics: {metrics}")
        return grid, metrics

# ─────────────────────────────────────────
# 4. TRAIN — RANDOM FOREST (run 2)
# ─────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test, cv, X_all, features):
    print("[4/6] Training Random Forest (Run 2)...")
    with mlflow.start_run(run_name="random-forest") as run:
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("phase", "classification")

        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf',   RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        params = {
            'clf__n_estimators':    [100, 200],
            'clf__max_depth':       [None, 5, 10],
            'clf__min_samples_split': [2, 5],
            'clf__max_features':    ['sqrt', 'log2']
        }
        grid = GridSearchCV(pipeline, params, cv=cv, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        y_prob = grid.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "f1_score":  round(f1_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        }
        best_params = {
            "n_estimators":      grid.best_params_['clf__n_estimators'],
            "max_depth":         str(grid.best_params_['clf__max_depth']),
            "min_samples_split": grid.best_params_['clf__min_samples_split'],
            "max_features":      grid.best_params_['clf__max_features'],
        }

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))
        mlflow.log_param("n_features", len(features))

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                               display_labels=['High Stock', 'Low Stock']).plot(ax=ax, colorbar=False)
        ax.set_title("RF — Confusion Matrix")
        plt.tight_layout()
        plt.savefig("/tmp/rf_cm.png", dpi=120)
        plt.close()
        mlflow.log_artifact("/tmp/rf_cm.png", artifact_path="plots")

        # ROC curve
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, y_prob, name="Random Forest", ax=ax)
        ax.plot([0,1],[0,1],'k--', linewidth=0.8)
        ax.set_title("RF — ROC Curve")
        plt.tight_layout()
        plt.savefig("/tmp/rf_roc.png", dpi=120)
        plt.close()
        mlflow.log_artifact("/tmp/rf_roc.png", artifact_path="plots")

        # Feature importance
        importances = grid.best_estimator_.named_steps['clf'].feature_importances_
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(7, 5))
        feat_imp.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title("RF — Feature Importance")
        plt.tight_layout()
        plt.savefig("/tmp/rf_importance.png", dpi=120)
        plt.close()
        mlflow.log_artifact("/tmp/rf_importance.png", artifact_path="plots")

        # Log model to registry
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            grid, "model",
            signature=signature,
            registered_model_name=MODEL_NAME
        )

        # Save model locally for API
        joblib.dump(grid, f"{MODEL_DIR}/rf_model.pkl")
        joblib.dump(features, f"{MODEL_DIR}/features.pkl")

        # Save feature list as artifact
        with open("/tmp/features.json", "w") as f:
            json.dump(features, f)
        mlflow.log_artifact("/tmp/features.json")

        run_id = run.info.run_id
        print(f"    RF metrics: {metrics}")
        print(f"    Run ID: {run_id}")
        return grid, metrics, run_id
    



# ─────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────
def main():
 try:
    print("\n" + "="*60)
    print("  MLOps Training Pipeline — Stock Classification")
    print("="*60 + "\n")

    # Load & preprocess
    df = load_data(DB_URL)
    X, y, features, encoders = preprocess(df)

    # Split
    print("[5/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    # Train both models (2 MLflow runs)
    lr_model,  lr_metrics              = train_logistic_regression(X_train, X_test, y_train, y_test, cv)
    rf_model,  rf_metrics, run_id      = train_random_forest(X_train, X_test, y_train, y_test, cv, X, features)

    # Compare
    print("\n[6/6] Final comparison:")
    print(f"    {'Metric':<12} {'LR':>8} {'RF':>8}")
    print(f"    {'-'*30}")
    for key in lr_metrics:
        print(f"    {key:<12} {lr_metrics[key]:>8} {rf_metrics[key]:>8}")

    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {MODEL_DIR}/rf_model.pkl")
    print(f"   MLflow run ID:  {run_id}")
    print(f"   View UI at:     {MLFLOW_URI}")

 except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
    print("\n✅ Training completed successfully!")
    print("Container will stay alive for 1 hour. Press Ctrl+C to stop.")
    import time
    time.sleep(3600)  # Keep container alive for 1 hour