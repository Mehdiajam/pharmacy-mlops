# ============================================================
# app.py — Streamlit Web App → FastAPI Integration
# ============================================================

import streamlit as st
import requests
import json
import pandas as pd

API_URL = "http://fastapi:8000"

st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
<style>
    .result-low  { background:#fff0f0; border-left:4px solid #dc3545; padding:16px; border-radius:8px; }
    .result-high { background:#f0fff4; border-left:4px solid #28a745; padding:16px; border-radius:8px; }
    .metric-box  { background:#f8f9fa; border-radius:8px; padding:12px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("📦 Pharmacy Stock Level Predictor")
st.markdown("Enter product details below to predict whether stock is **Low** or **High**.")
st.markdown("---")

# ─────────────────────────────────────────
# API HEALTH CHECK
# ─────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health["model_loaded"]:
        st.success("✅ API connected — Model loaded and ready")
    else:
        st.warning("⚠️ API connected but model not loaded. Run the training pipeline first.")
except Exception:
    st.error("❌ Cannot connect to API. Make sure the FastAPI container is running.")
    st.stop()

st.markdown("---")

# ─────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────
st.subheader("Product Details")

col1, col2, col3 = st.columns(3)

with col1:
    prix_achat  = st.number_input("Purchase Price HT (DT)",  min_value=0.0, value=25.5,  step=0.1)
    prix_vente  = st.number_input("Selling Price HT (DT)",   min_value=0.0, value=35.0,  step=0.1)
    prix_ttc    = st.number_input("Selling Price TTC (DT)",  min_value=0.0, value=41.3,  step=0.1)

with col2:
    famille     = st.selectbox("Product Family (Famille)", [
        "TOILETTE", "COMPRIME", "COMP. ALIM.", "OPH-ORL",
        "POMMADE", "SUPPOSITOIRE", "DIVERS", "NATURE"
    ])
    designation = st.text_input("Designation", value="SHAMPOO")
    libelle     = st.text_input("Libelle",      value="SHAMPOO 200ML")

with col3:
    ville               = st.text_input("City (Ville)", value="Tunis")
    stock_duration_days = st.number_input("Stock Duration (days)", min_value=0.0, value=30.0, step=1.0)

# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Predict Stock Level", type="primary", use_container_width=True)

if predict_btn:
    payload = {
        "Prix_d_achat_HT":    prix_achat,
        "Prix_de_vente_HT":   prix_vente,
        "Prix_de_vente_TTC":  prix_ttc,
        "Famille":             famille,
        "Designation":         designation,
        "Libelle":             libelle,
        "Ville":               ville,
        "Stock_Duration_Days": stock_duration_days
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            st.markdown("---")
            st.subheader("Prediction Result")

            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            with col_res1:
                st.metric("Prediction", result["label"])
            with col_res2:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            with col_res3:
                st.metric("P(Low Stock)",  f"{result['probability_low']*100:.1f}%")
            with col_res4:
                st.metric("P(High Stock)", f"{result['probability_high']*100:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)
            if result["prediction"] == 1:
                st.markdown(f"""
                <div class="result-low">
                    <h3>🔴 LOW STOCK ALERT</h3>
                    <p>This product is predicted to be <b>below the reorder threshold</b>.</p>
                    <p>Confidence: <b>{result['confidence']*100:.1f}%</b></p>
                    <p><i>Recommended action: Trigger a reorder immediately.</i></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-high">
                    <h3>🟢 HIGH STOCK</h3>
                    <p>This product has <b>adequate stock levels</b>.</p>
                    <p>Confidence: <b>{result['confidence']*100:.1f}%</b></p>
                    <p><i>No immediate reorder action needed.</i></p>
                </div>
                """, unsafe_allow_html=True)

            # Show raw JSON
            with st.expander("Raw API Response"):
                st.json(result)

        except requests.exceptions.HTTPError as e:
            st.error(f"API Error: {e.response.json().get('detail', str(e))}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

# ─────────────────────────────────────────
# SIDEBAR — MODEL INFO
# ─────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    try:
        info = requests.get(f"{API_URL}/model/info", timeout=3).json()
        st.write(f"**Type:** {info['model_type']}")
        st.write(f"**Features:** {info['n_features']}")
        with st.expander("Feature list"):
            st.write(info['features'])
    except:
        st.write("Model info unavailable")

    st.markdown("---")
    st.header("🔗 Links")
    st.markdown("- [MLflow UI](http://localhost:5000)")
    st.markdown("- [API Docs](http://localhost:8000/docs)")
    st.markdown("- [n8n](http://localhost:5678)")