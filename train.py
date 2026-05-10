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
from urllib.parse import quote, unquote, urlparse, urlunparse, parse_qsl

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay,
                             mean_squared_error, mean_absolute_error, r2_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import joblib

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
def normalize_db_url(db_url):
    if isinstance(db_url, bytes):
        for encoding in ("utf-8", "cp1252", "latin1"):
            try:
                db_url = db_url.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not decode DB_URL bytes using utf-8, cp1252, or latin1")

    # Now db_url is str
    try:
        db_url.encode('utf-8')
    except UnicodeEncodeError:
        # Contains non-UTF-8 chars, assume latin1 encoding
        db_url = db_url.encode('latin1').decode('utf-8')

    parsed = urlparse(db_url)
    if parsed.scheme in ("postgresql", "postgres"):
        username = quote(unquote(parsed.username)) if parsed.username else None
        password = quote(unquote(parsed.password)) if parsed.password else None
        netloc = ""
        if username:
            netloc += username
        if password is not None:
            netloc += f":{password}"
        if netloc:
            netloc += "@"
        if parsed.hostname:
            netloc += parsed.hostname
            if parsed.port:
                netloc += f":{parsed.port}"

        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if "client_encoding" not in query:
            query["client_encoding"] = "utf8"
        query_str = "&".join(f"{quote(str(k))}={quote(str(v))}" for k, v in query.items())

        db_url = urlunparse((parsed.scheme, netloc, parsed.path or "", parsed.params, query_str, parsed.fragment))

    return db_url

DB_URL       = normalize_db_url(os.getenv("DB_URL", "postgresql://postgres:@host.docker.internal:5432/parapharmacie_BD"))
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
    engine = create_engine(normalize_db_url(db_url), connect_args={"options": "-c client_encoding=UTF8"})
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
        import tempfile
        temp_dir = tempfile.gettempdir()
        cm_path = os.path.join(temp_dir, "lr_cm.png")
        plt.savefig(cm_path, dpi=120)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

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
        import tempfile
        temp_dir = tempfile.gettempdir()
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                               display_labels=['High Stock', 'Low Stock']).plot(ax=ax, colorbar=False)
        ax.set_title("RF — Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(temp_dir, "rf_cm.png")
        plt.savefig(cm_path, dpi=120)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # ROC curve
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, y_prob, name="Random Forest", ax=ax)
        ax.plot([0,1],[0,1],'k--', linewidth=0.8)
        ax.set_title("RF — ROC Curve")
        plt.tight_layout()
        roc_path = os.path.join(temp_dir, "rf_roc.png")
        plt.savefig(roc_path, dpi=120)
        plt.close()
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Feature importance
        importances = grid.best_estimator_.named_steps['clf'].feature_importances_
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(7, 5))
        feat_imp.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title("RF — Feature Importance")
        plt.tight_layout()
        imp_path = os.path.join(temp_dir, "rf_importance.png")
        plt.savefig(imp_path, dpi=120)
        plt.close()
        mlflow.log_artifact(imp_path, artifact_path="plots")

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
        import tempfile
        temp_dir = tempfile.gettempdir()
        features_path = os.path.join(temp_dir, "features.json")
        with open(features_path, "w") as f:
            json.dump(features, f)
        mlflow.log_artifact(features_path)

        run_id = run.info.run_id
        print(f"    RF metrics: {metrics}")
        print(f"    Run ID: {run_id}")
        return grid, metrics, run_id
    



# ─────────────────────────────────────────
# 5. TIME SERIES FORECASTING
# ─────────────────────────────────────────
def train_time_series_forecasting(df: pd.DataFrame):
    print("[5/6] Training Time Series Forecasting Models...")
    
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller
        from prophet import Prophet
        import itertools
        import tempfile
    except ImportError as e:
        print(f"    ⚠️  Skipping forecasting (missing dependency: {e}). Continue with classification only.")
        return None, None, None
    
    with mlflow.start_run(run_name="time-series-forecasting"):
        mlflow.set_tag("model_type", "TimeSeries")
        mlflow.set_tag("phase", "forecasting")

        # 1. BUILD MONTHLY TIME SERIES
        # Clean data: ensure Date_In is valid and drop NaT
        df_ts = df.dropna(subset=['Date_In', 'Quantite']).copy()
        ts_raw = df_ts.groupby('Date_In')['Quantite'].sum().sort_index()
        ts_monthly = ts_raw.resample('MS').sum()
        ts_monthly = ts_monthly.asfreq('MS', fill_value=0)

        print(f"    Time series length: {len(ts_monthly)} months")
        print(f"    Date range: {ts_monthly.index.min()} → {ts_monthly.index.max()}")

        # Skip if insufficient data
        if len(ts_monthly) < 12:
            print(f"    ⚠️  Insufficient data ({len(ts_monthly)} points < 12 required). Skipping forecasting.")
            return None, None, None

        # 2. STATIONARITY CHECK
        adf_result = adfuller(ts_monthly.dropna())
        is_stationary = adf_result[1] < 0.05
        print(f"    ADF test: p={adf_result[1]:.4f} → {'STATIONARY' if is_stationary else 'NON-STATIONARY'}")

        # 3. TRAIN-TEST SPLIT (time-aware)
        n_test = min(6, len(ts_monthly) // 4)  # Hold out last 6 months or 1/4 of data
        train = ts_monthly.iloc[:-n_test]
        test = ts_monthly.iloc[-n_test:]

        print(f"    Train: {len(train)} months | Test: {len(test)} months")

        # 4. SARIMA GRID SEARCH
        print("    Searching SARIMA parameters...")
        best_aic = np.inf
        best_order = (1, 0, 1)
        
        for p, d, q in itertools.product(range(0, 3), range(0, 2), range(0, 3)):
            try:
                model = SARIMAX(train, order=(p, d, q), seasonal_order=(1, 1, 1, 12),
                               enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
            except:
                continue

        print(f"    Best SARIMA order: {best_order}")
        
        sarima_model = SARIMAX(train, order=best_order, seasonal_order=(1, 1, 1, 12),
                              enforce_stationarity=False, enforce_invertibility=False)
        sarima_result = sarima_model.fit(disp=False)
        sarima_forecast = sarima_result.forecast(steps=n_test)
        sarima_forecast.index = test.index

        # 5. PROPHET
        print("    Training Prophet model...")
        prophet_train = pd.DataFrame({'ds': train.index, 'y': train.values})
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                               daily_seasonality=False, interval_width=0.95)
        prophet_model.fit(prophet_train)
        
        future = prophet_model.make_future_dataframe(periods=n_test, freq='MS')
        prophet_forecast_full = prophet_model.predict(future)
        prophet_forecast = prophet_forecast_full.set_index('ds')['yhat'].iloc[-n_test:]
        prophet_forecast.index = test.index

        # 6. EVALUATION
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        sarima_rmse = np.sqrt(mean_squared_error(test.values, sarima_forecast.values))
        sarima_mae = mean_absolute_error(test.values, sarima_forecast.values)
        prophet_rmse = np.sqrt(mean_squared_error(test.values, prophet_forecast.values))
        prophet_mae = mean_absolute_error(test.values, prophet_forecast.values)

        metrics_ts = {
            "sarima_rmse": round(sarima_rmse, 2),
            "sarima_mae": round(sarima_mae, 2),
            "prophet_rmse": round(prophet_rmse, 2),
            "prophet_mae": round(prophet_mae, 2),
            "train_points": len(train),
            "test_points": len(test),
            "is_stationary": is_stationary
        }

        mlflow.log_params({"sarima_order": str(best_order), "prophet_seasonality": "yearly"})
        mlflow.log_metrics(metrics_ts)

        # 7. SAVE MODELS
        joblib.dump(sarima_result, f"{MODEL_DIR}/sarima_model.pkl")
        joblib.dump(prophet_model, f"{MODEL_DIR}/prophet_model.pkl")
        joblib.dump({'train': train, 'test': test}, f"{MODEL_DIR}/ts_data.pkl")

        # 8. VISUALIZE
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(train.index, train.values, label='Train', linewidth=1.5, color='steelblue')
        axes[0].plot(test.index, test.values, label='Actual', linewidth=2, marker='o', color='black')
        axes[0].plot(sarima_forecast.index, sarima_forecast.values, label='SARIMA', 
                    linewidth=2, linestyle='--', marker='s', color='tomato')
        axes[0].set_title('SARIMA Forecast')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(train.index, train.values, label='Train', linewidth=1.5, color='steelblue')
        axes[1].plot(test.index, test.values, label='Actual', linewidth=2, marker='o', color='black')
        axes[1].plot(prophet_forecast.index, prophet_forecast.values, label='Prophet',
                    linewidth=2, linestyle='--', marker='s', color='darkorange')
        axes[1].set_title('Prophet Forecast')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        
        temp_plot_path = os.path.join(tempfile.gettempdir(), "ts_forecast.png")
        plt.savefig(temp_plot_path, dpi=120)
        plt.close()
        mlflow.log_artifact(temp_plot_path, artifact_path="plots")

        print(f"    ✅ Forecasting metrics: SARIMA RMSE={metrics_ts['sarima_rmse']}, Prophet RMSE={metrics_ts['prophet_rmse']}")
        
        return sarima_result, prophet_model, metrics_ts


# ─────────────────────────────────────────
# 5b. TRAIN — PROFITABILITY REGRESSOR
# ─────────────────────────────────────────
def train_profitability_regressor(X_train, X_test, y_train_reg, y_test_reg, features):
    print("[5b/6] Training Profitability Regressor (Run 3)...")
    with mlflow.start_run(run_name="profitability-regressor") as run:
        mlflow.set_tag("model_type", "RandomForestRegressor")
        mlflow.set_tag("phase", "regression")

        # Regressor for quantity
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train_reg)

        y_pred = grid.predict(X_test)

        metrics = {
            "mae":  round(mean_absolute_error(y_test_reg, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test_reg, y_pred)), 4),
            "r2":   round(r2_score(y_test_reg, y_pred), 4),
        }

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)

        # Log model
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            grid, "model",
            signature=signature,
            registered_model_name="profitability-regressor"
        )

        # Save model locally for API
        joblib.dump(grid, f"{MODEL_DIR}/profit_model.pkl")

        print(f"    Regressor metrics: {metrics}")
        return grid, metrics


# ─────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────
def main():
    try:
        print("\n" + "="*60)
        print("  MLOps Training Pipeline — Stock Classification + Forecasting")
        print("="*60 + "\n")

        # Load & preprocess
        df = load_data(DB_URL)
        X, y, features, encoders = preprocess(df)

        # Split for classification (y is Stock_Level)
        print("[5/6] Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Regression target (y_reg is Quantite)
        y_reg = df.loc[X.index, 'Quantite'].copy()
        y_train_reg = y_reg.loc[X_train.index]
        y_test_reg = y_reg.loc[X_test.index]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

        # Train classification models
        lr_model,  lr_metrics              = train_logistic_regression(X_train, X_test, y_train, y_test, cv)
        rf_model,  rf_metrics, run_id      = train_random_forest(X_train, X_test, y_train, y_test, cv, X, features)

        # Train profitability regressor
        pr_model, pr_metrics = train_profitability_regressor(X_train, X_test, y_train_reg, y_test_reg, features)

        # Save encoders for consistent inference
        joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
        print(f"    ✅ Encoders saved to {MODEL_DIR}/encoders.pkl")

        # Train time series forecasting
        sarima_model, prophet_model, ts_metrics = train_time_series_forecasting(df)

        # Comparison summary
        print("\n[6/6] Classification Model Comparison:")
        print(f"    {'Metric':<12} {'LR':>8} {'RF':>8}")
        print(f"    {'-'*30}")
        for key in lr_metrics:
            print(f"    {key:<12} {lr_metrics[key]:>8} {rf_metrics[key]:>8}")

        print("\n[6/6] Profitability Regression Metrics:")
        for key, value in pr_metrics.items():
            print(f"    {key:<12} {value:>8}")

        if ts_metrics:
            print("\n[6/6] Time Series Forecasting Metrics:")
            for key, value in ts_metrics.items():
                print(f"    {key:<20} {value}")

        print(f"\n✅ Training complete!")
        print(f"   Classification model saved to: {MODEL_DIR}/rf_model.pkl")
        print(f"   Regression model saved to:     {MODEL_DIR}/profit_model.pkl")
        print(f"   Forecasting models saved to: {MODEL_DIR}/sarima_model.pkl, {MODEL_DIR}/prophet_model.pkl")
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