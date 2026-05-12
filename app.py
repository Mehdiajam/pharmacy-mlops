# ============================================================
# app.py — Paratech MLOps Dashboard
# ============================================================

import streamlit as st
import requests
import json
import pandas as pd
import streamlit.components.v1 as components
import time

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
API_URL = "http://fastapi:8000"

USER_DASHBOARDS = {
    "manager": {
        "password": "manager123",
        "name": "Head Office Admin",
        "role": "Administrator",
        "avatar": "🏢",
        "url": "https://app.powerbi.com/reportEmbed?reportId=d94d6e3c-0c8e-405d-b457-1debc53c5539&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730"
    },
    "mehdi": {
        "password": "mehdi123",
        "name": "Stock Manager",
        "role": "Inventory Specialist",
        "avatar": "📦",
        "url": "https://app.powerbi.com/reportEmbed?reportId=408f00aa-8560-4180-9906-175f960ea7ac&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730&pageName=Stock+Manager"
    },
    "hechmi": {
        "password": "hechmi123",
        "name": "Financial Manager",
        "role": "Finance & ROI",
        "avatar": "💰",
        "url": "https://app.powerbi.com/reportEmbed?reportId=d94d6e3c-0c8e-405d-b457-1debc53c5539&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730&pageName=financial+Overview"
    },
    "hedi": {
        "password": "hedi123",
        "name": "Marketing Manager",
        "role": "Growth & Analytics",
        "avatar": "📊",
        "url": "https://app.powerbi.com/reportEmbed?reportId=65ca80b5-adaf-448f-a122-b4efdb259eae&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730"
    }
}

st.set_page_config(
    page_title="Paratech — Smart Pharmacy Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'show_login' not in st.session_state:
    st.session_state.show_login = False

# ─────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&display=swap');

:root {
    --green-900: #052e16;
    --green-800: #166534;
    --green-700: #15803d;
    --green-600: #16a34a;
    --green-500: #22c55e;
    --green-400: #4ade80;
    --green-100: #dcfce7;
    --green-50:  #f0fdf4;
    --sand:      #faf7f2;
    --sand-dark: #f5f0e8;
    --ink:       #0f1a0f;
    --ink-light: #374151;
    --border:    #e2e8e2;
    --white:     #ffffff;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow:    0 4px 16px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04);
    --shadow-lg: 0 20px 40px rgba(0,0,0,0.12), 0 4px 12px rgba(0,0,0,0.06);
    --radius:    14px;
    --radius-sm: 8px;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--sand);
    color: var(--ink);
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--green-800) !important;
    font-weight: 400 !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: var(--sand);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Buttons */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    transition: all 0.18s ease !important;
    letter-spacing: -0.01em !important;
}
.stButton > button[kind="primary"] {
    background: var(--green-700) !important;
    color: white !important;
    border-color: var(--green-700) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--green-800) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow) !important;
}
.stButton > button:hover {
    border-color: var(--green-500) !important;
    color: var(--green-700) !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    font-family: 'DM Sans', sans-serif !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    background: var(--white) !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--green-500) !important;
    box-shadow: 0 0 0 3px rgba(34,197,94,0.12) !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    box-shadow: var(--shadow-sm);
}
[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--ink-light) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    color: var(--green-800) !important;
    font-size: 1.8rem !important;
}

/* Form */
[data-testid="stForm"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius) !important;
    padding: 2rem !important;
    box-shadow: var(--shadow) !important;
}

/* Cards */
.paratech-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s, transform 0.2s;
}
.paratech-card:hover {
    box-shadow: var(--shadow);
    transform: translateY(-2px);
}

/* Alerts */
.alert-low {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #ef4444;
    border-radius: var(--radius);
    padding: 20px 24px;
}
.alert-high {
    background: var(--green-50);
    border: 1px solid var(--green-100);
    border-left: 4px solid var(--green-500);
    border-radius: var(--radius);
    padding: 20px 24px;
}
.alert-info {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6;
    border-radius: var(--radius);
    padding: 20px 24px;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* Pill badge */
.pill {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.pill-green { background: var(--green-100); color: var(--green-800); }
.pill-red   { background: #fee2e2; color: #991b1b; }
.pill-blue  { background: #dbeafe; color: #1e40af; }
.pill-gray  { background: #f3f4f6; color: #374151; }

/* Status dot */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-green { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.dot-red   { background: #ef4444; }
.dot-yellow{ background: #f59e0b; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HELPER COMPONENTS
# ─────────────────────────────────────────
def card(content, padding="24px"):
    st.markdown(f"""
    <div class="paratech-card" style="padding:{padding}">
        {content}
    </div>
    """, unsafe_allow_html=True)

def section_header(title, subtitle=None, icon=None):
    icon_html = f"<span style='margin-right:10px'>{icon}</span>" if icon else ""
    sub_html = f"<p style='font-family:DM Sans,sans-serif;color:#6b7280;font-size:1rem;margin:4px 0 0 0;font-weight:400'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='margin-bottom:1.5rem'>
        <h2 style='margin:0;font-size:1.8rem'>{icon_html}{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def api_status():
    try:
        health = requests.get(f"{API_URL}/health", timeout=1).json()
        if health.get("model_loaded"):
            return "online"
        return "loading"
    except:
        return "offline"

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.page = 'home'
    st.session_state.show_login = False
    st.rerun()

# ─────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────
def show_landing():
    st.markdown("""
    <style>
    .hero-section {
        background: linear-gradient(135deg, #052e16 0%, #166534 50%, #15803d 100%);
        border-radius: 24px;
        padding: 80px 60px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(74,222,128,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-section::after {
        content: '';
        position: absolute;
        bottom: -80px; left: -40px;
        width: 240px; height: 240px;
        background: radial-gradient(circle, rgba(34,197,94,0.10) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3.8rem;
        color: #ffffff !important;
        line-height: 1.1;
        margin: 0 0 12px 0;
        font-weight: 400 !important;
        letter-spacing: -0.02em;
    }
    .hero-tag {
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4ade80;
        margin-bottom: 20px;
    }
    .hero-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.15rem;
        color: rgba(255,255,255,0.80);
        max-width: 520px;
        line-height: 1.7;
        margin: 0;
        font-weight: 300;
    }
    .hero-stats {
        display: flex;
        gap: 40px;
        margin-top: 48px;
    }
    .hero-stat-val {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #ffffff;
        display: block;
        line-height: 1;
    }
    .hero-stat-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 4px;
        display: block;
    }
    .feature-card {
        background: white;
        border: 1px solid #e2e8e2;
        border-radius: 16px;
        padding: 28px;
        height: 100%;
        transition: all 0.2s;
        position: relative;
        overflow: hidden;
    }
    .feature-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #22c55e, #16a34a);
        transform: scaleX(0);
        transition: transform 0.2s;
        transform-origin: left;
    }
    .feature-card:hover { box-shadow: 0 8px 32px rgba(0,0,0,0.10); transform: translateY(-3px); }
    .feature-card:hover::after { transform: scaleX(1); }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 16px;
        display: block;
    }
    .feature-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.2rem;
        color: #166534 !important;
        margin: 0 0 10px 0;
        font-weight: 400 !important;
    }
    .feature-desc {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        color: #6b7280;
        line-height: 1.65;
        margin: 0;
    }
    .how-step {
        display: flex;
        align-items: flex-start;
        gap: 20px;
        padding: 24px;
        background: white;
        border-radius: 14px;
        border: 1px solid #e2e8e2;
        margin-bottom: 12px;
    }
    .step-num {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #166534, #22c55e);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'DM Serif Display', serif;
        font-size: 1rem;
        color: white;
        flex-shrink: 0;
    }
    .step-title {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #166534;
        margin: 0 0 4px 0;
    }
    .step-desc {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: #6b7280;
        margin: 0;
        line-height: 1.55;
    }
    .cta-box {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1px solid #bbf7d0;
        border-radius: 20px;
        padding: 48px;
        text-align: center;
        margin-top: 3rem;
    }
    .testimonial {
        background: white;
        border: 1px solid #e2e8e2;
        border-radius: 14px;
        padding: 24px;
        position: relative;
    }
    .testimonial::before {
        content: '"';
        font-family: 'DM Serif Display', serif;
        font-size: 4rem;
        color: #dcfce7;
        position: absolute;
        top: 8px; left: 16px;
        line-height: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── HERO ──
    st.markdown("""
    <div class="hero-section">
        <div class="hero-tag">🌿 Parapharmacy Intelligence Platform</div>
        <h1 class="hero-title">Smart inventory.<br>Better profits.<br>Paratech.</h1>
        <p class="hero-subtitle">
            The AI-powered platform designed exclusively for parapharmacies — monitor stock levels in real time, predict shortages before they happen, and unlock data-driven insights to grow your business.
        </p>
        <div class="hero-stats">
            <div>
                <span class="hero-stat-val">86.1%</span>
                <span class="hero-stat-label">Prediction accuracy</span>
            </div>
            <div>
                <span class="hero-stat-val">92.1%</span>
                <span class="hero-stat-label">Recommendation precision</span>
            </div>
            <div>
                <span class="hero-stat-val">3.3%</span>
                <span class="hero-stat-label">Anomalies detected</span>
            </div>
            <div>
                <span class="hero-stat-val">6</span>
                <span class="hero-stat-label">ML models deployed</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FEATURES ──
    st.markdown("""
    <div style='text-align:center;margin-bottom:2rem'>
        <div style='font-family:DM Sans;font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#16a34a;margin-bottom:10px'>What Paratech does</div>
        <h2 style='font-family:DM Serif Display,serif;font-size:2.2rem;color:#166534!important;font-weight:400!important;margin:0'>Everything your parapharmacy needs</h2>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <h3 class="feature-title">AI Stock Prediction</h3>
            <p class="feature-desc">Our Random Forest model predicts with 86.1% F1-score whether any product is heading toward a Low Stock situation — before it runs out. Get alerts before customers leave empty-handed.</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <h3 class="feature-title">Business Intelligence</h3>
            <p class="feature-desc">Role-based Power BI dashboards give each team member — stock managers, finance, marketing — exactly the insights they need. No information overload, no missing context.</p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">💰</span>
            <h3 class="feature-title">Profit Optimization</h3>
            <p class="feature-desc">Simulate pricing scenarios and get AI-powered profit projections for any product. Identify your strategic assets, volume drivers, and underperformers with one click.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f4, f5, f6 = st.columns(3)
    with f4:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🔍</span>
            <h3 class="feature-title">Anomaly Detection</h3>
            <p class="feature-desc">Isolation Forest and LOF algorithms continuously scan your inventory for unusual patterns — zero-margin products, extreme quantities, overpriced zero-stock items — and flag them for review.</p>
        </div>
        """, unsafe_allow_html=True)
    with f5:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <h3 class="feature-title">Product Recommendations</h3>
            <p class="feature-desc">When a product is out of stock, Paratech instantly recommends the most similar alternatives with 92.1% within-category precision — keeping your sales moving.</p>
        </div>
        """, unsafe_allow_html=True)
    with f6:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">⏱</span>
            <h3 class="feature-title">Demand Forecasting</h3>
            <p class="feature-desc">Prophet and SARIMA time-series models forecast monthly stock quantities, helping you plan procurement cycles and avoid both overstock and stockout situations.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── HOW IT WORKS ──
    col_how, col_metrics = st.columns([1.1, 0.9])
    with col_how:
        st.markdown("""
        <div style='margin-bottom:1.5rem'>
            <div style='font-family:DM Sans;font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#16a34a;margin-bottom:10px'>How it works</div>
            <h2 style='font-family:DM Serif Display,serif;font-size:1.8rem;color:#166534!important;font-weight:400!important;margin:0'>From data to decision<br>in seconds</h2>
        </div>
        <div class="how-step">
            <div class="step-num">1</div>
            <div>
                <div class="step-title">Connect your inventory data</div>
                <div class="step-desc">Paratech integrates with your existing parapharmacy data warehouse via a star schema — no manual entry required.</div>
            </div>
        </div>
        <div class="how-step">
            <div class="step-num">2</div>
            <div>
                <div class="step-title">AI models train on your catalog</div>
                <div class="step-desc">6 ML models (classification, regression, clustering, forecasting, anomaly detection, recommendations) learn the patterns in your product and stock data.</div>
            </div>
        </div>
        <div class="how-step">
            <div class="step-num">3</div>
            <div>
                <div class="step-title">Get real-time predictions</div>
                <div class="step-desc">Enter any product's pricing and attributes — Paratech instantly predicts stock risk, estimates profit potential, and recommends alternatives.</div>
            </div>
        </div>
        <div class="how-step">
            <div class="step-num">4</div>
            <div>
                <div class="step-title">Monitor and improve</div>
                <div class="step-desc">Prometheus and Grafana track model health, data drift, and system performance in real time — with automatic retraining triggers.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown("""
        <div style='margin-bottom:1.5rem'>
            <div style='font-family:DM Sans;font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#16a34a;margin-bottom:10px'>Proven results</div>
            <h2 style='font-family:DM Serif Display,serif;font-size:1.8rem;color:#166534!important;font-weight:400!important;margin:0'>ML performance<br>by the numbers</h2>
        </div>
        """, unsafe_allow_html=True)

        metrics = [
            ("Classification", "Random Forest", "F1 Score", "0.861", "green"),
            ("Regression",     "XGBoost",       "R² Score", "0.300", "blue"),
            ("Clustering",     "K-Means k=2",   "Silhouette","0.502","green"),
            ("Forecasting",    "Prophet",        "RMSE",     "179.3","blue"),
            ("Anomaly Det.",   "IF + LOF",       "Detected", "7 (3.3%)","green"),
            ("Recommender",   "Cosine Sim.",    "Precision","92.1%","green"),
        ]
        for task, model, metric_name, value, color in metrics:
            pill_class = f"pill-{'green' if color == 'green' else 'blue'}"
            st.markdown(f"""
            <div style='background:white;border:1px solid #e2e8e2;border-radius:12px;padding:14px 18px;margin-bottom:10px;display:flex;align-items:center;justify-content:space-between'>
                <div>
                    <div style='font-family:DM Sans;font-weight:600;font-size:0.88rem;color:#166534'>{task}</div>
                    <div style='font-family:DM Sans;font-size:0.78rem;color:#9ca3af;margin-top:2px'>{model} · {metric_name}</div>
                </div>
                <div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#166534'>{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── TESTIMONIALS ──
    st.markdown("""
    <div style='text-align:center;margin-bottom:2rem'>
        <h2 style='font-family:DM Serif Display,serif;font-size:2rem;color:#166534!important;font-weight:400!important;margin:0'>Built for parapharmacies,<br>by data scientists</h2>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)
    testimonials = [
        ("Stock Manager", "The Low Stock alerts have completely changed how we manage reorders. We haven't had a stockout in months.", "📦"),
        ("Financial Manager", "The profit simulation tool helped us identify 4 underperforming product lines we were overinvesting in.", "💰"),
        ("Marketing Manager", "Having my own dashboard with exactly the data I need — without filtering through irrelevant numbers — is a game changer.", "📊"),
    ]
    for col, (role, text, icon) in zip([t1, t2, t3], testimonials):
        with col:
            st.markdown(f"""
            <div class="testimonial">
                <div style='padding-top:28px;font-family:DM Sans;font-size:0.9rem;color:#374151;line-height:1.65;font-style:italic'>{text}</div>
                <div style='margin-top:16px;display:flex;align-items:center;gap:10px'>
                    <span style='font-size:1.4rem'>{icon}</span>
                    <div style='font-family:DM Sans;font-weight:600;font-size:0.82rem;color:#166534'>{role}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── CTA ──
    st.markdown("""
    <div class="cta-box">
        <div style='font-family:DM Sans;font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#16a34a;margin-bottom:12px'>Ready to get started?</div>
        <h2 style='font-family:DM Serif Display,serif;font-size:2.2rem;color:#166534!important;font-weight:400!important;margin:0 0 12px 0'>Sign in to your Paratech workspace</h2>
        <p style='font-family:DM Sans;color:#6b7280;font-size:1rem;margin:0 auto;max-width:460px;line-height:1.65'>Each team member gets a personalized dashboard with role-specific insights and AI tools tailored to their responsibilities.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, cta_col, _ = st.columns([2, 1, 2])
    with cta_col:
        if st.button("Sign In to Paratech →", type="primary", use_container_width=True):
            st.session_state.show_login = True
            st.rerun()

    st.markdown("""
    <div style='text-align:center;padding:40px 0 10px 0;font-family:DM Sans;font-size:0.8rem;color:#9ca3af'>
        © 2026 Paratech · Smart Pharmacy Intelligence · Built with ❤️ for parapharmacies
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────
def show_login():
    st.markdown("<br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        # Back to landing
        if st.button("← Back to Paratech", type="secondary"):
            st.session_state.show_login = False
            st.rerun()

        st.markdown("""
        <div style='text-align:center;padding:32px 0 24px 0'>
            <div style='font-size:3rem;margin-bottom:12px'>🌿</div>
            <h2 style='font-family:DM Serif Display,serif;font-size:2rem;color:#166534!important;font-weight:400!important;margin:0'>Welcome back</h2>
            <p style='font-family:DM Sans;color:#6b7280;margin:6px 0 0 0;font-size:0.95rem'>Sign in to your Paratech workspace</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            user = st.text_input("Username", placeholder="Enter your username").lower().strip()
            pwd  = st.text_input("Password", type="password", placeholder="••••••••")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("Sign In →", use_container_width=True)

            if submit:
                if user in USER_DASHBOARDS and USER_DASHBOARDS[user]["password"] == pwd:
                    st.session_state.authenticated = True
                    st.session_state.username = user
                    st.session_state.page = 'prediction' if user in ['manager', 'mehdi'] else 'dashboard'
                    st.success(f"Welcome back, {USER_DASHBOARDS[user]['name']}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please check your username and password.")

        st.markdown("""
        <div style='margin-top:20px;background:#f0fdf4;border:1px solid #dcfce7;border-radius:10px;padding:14px 18px'>
            <div style='font-family:DM Sans;font-size:0.8rem;font-weight:600;color:#166534;margin-bottom:8px'>Available accounts</div>
            <div style='font-family:DM Sans;font-size:0.78rem;color:#6b7280;line-height:1.8'>
                manager / manager123 &nbsp;·&nbsp; mehdi / mehdi123<br>
                hechmi / hechmi123 &nbsp;·&nbsp; hedi / hedi123
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR (authenticated)
# ─────────────────────────────────────────
def show_sidebar():
    user = st.session_state.username
    info = USER_DASHBOARDS[user]
    status = api_status()
    status_html = {
        "online":  "<span class='status-dot dot-green'></span><span style='font-size:0.78rem;font-weight:600;color:#166534'>ML Engine Online</span>",
        "loading": "<span class='status-dot dot-yellow'></span><span style='font-size:0.78rem;font-weight:600;color:#92400e'>Engine Loading...</span>",
        "offline": "<span class='status-dot dot-red'></span><span style='font-size:0.78rem;font-weight:600;color:#991b1b'>API Offline</span>",
    }[status]

    with st.sidebar:
        # Logo
        st.markdown("""
        <div style='display:flex;align-items:center;gap:10px;padding:4px 0 20px 0'>
            <span style='font-size:1.6rem'>🌿</span>
            <div>
                <div style='font-family:DM Serif Display,serif;font-size:1.3rem;color:#166534;line-height:1'>Paratech</div>
                <div style='font-family:DM Sans;font-size:0.7rem;color:#9ca3af;letter-spacing:0.04em;text-transform:uppercase'>Intelligence Platform</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # User card
        st.markdown(f"""
        <div style='background:#f0fdf4;border:1px solid #dcfce7;border-radius:12px;padding:14px 16px;margin-bottom:20px'>
            <div style='display:flex;align-items:center;gap:12px'>
                <div style='font-size:1.8rem;line-height:1'>{info['avatar']}</div>
                <div>
                    <div style='font-family:DM Sans;font-weight:600;font-size:0.9rem;color:#166534'>{info['name']}</div>
                    <div style='font-family:DM Sans;font-size:0.75rem;color:#6b7280;margin-top:2px'>{info['role']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-family:DM Sans;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:10px'>Navigation</div>", unsafe_allow_html=True)

        # Navigation
        if user in ['manager', 'mehdi']:
            if st.button("🧠  Stock Prediction", use_container_width=True,
                         type="primary" if st.session_state.page == 'prediction' else "secondary"):
                st.session_state.page = 'prediction'; st.rerun()

        if st.button("📊  Business Analytics", use_container_width=True,
                     type="primary" if st.session_state.page == 'dashboard' else "secondary"):
            st.session_state.page = 'dashboard'; st.rerun()

        if user in ['manager', 'hechmi', 'hedi']:
            if st.button("💰  Profit Analysis", use_container_width=True,
                         type="primary" if st.session_state.page == 'profitability' else "secondary"):
                st.session_state.page = 'profitability'; st.rerun()

        if user == 'manager':
            if st.button("⏱  Time Forecasting", use_container_width=True,
                         type="primary" if st.session_state.page == 'forecasting' else "secondary"):
                st.session_state.page = 'forecasting'; st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Sans;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:10px'>System</div>", unsafe_allow_html=True)

        # API status
        st.markdown(f"""
        <div style='background:white;border:1px solid #e2e8e2;border-radius:10px;padding:10px 14px;margin-bottom:12px'>
            {status_html}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪  Sign Out", use_container_width=True):
            logout()


# ─────────────────────────────────────────
# PAGE: PREDICTION
# ─────────────────────────────────────────
def page_prediction():
    section_header("Stock Prediction", "AI-powered Low/High stock classification using Random Forest", "🧠")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='font-family:DM Sans;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:12px'>Pricing Structure</div>
        """, unsafe_allow_html=True)
        prix_achat = st.number_input("Purchase Price HT (DT)", value=25.5, step=0.1)
        prix_vente = st.number_input("Selling Price HT (DT)",  value=35.0, step=0.1)
        prix_ttc   = st.number_input("Selling Price TTC (DT)", value=41.3, step=0.1)

    with col2:
        st.markdown("""
        <div style='font-family:DM Sans;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:12px'>Product Attributes</div>
        """, unsafe_allow_html=True)
        famille     = st.selectbox("Product Family", ["TOILETTE","COMPRIME","COMP. ALIM.","OPH-ORL","POMMADE","SUPPOSITOIRE","DIVERS","NATURE"])
        designation = st.text_input("Designation", value="SHAMPOO")
        ville       = st.text_input("City Location", value="Tunis")

    # Model info banner
    st.markdown("""
    <div style='background:#f0fdf4;border:1px solid #dcfce7;border-radius:10px;padding:12px 18px;margin:16px 0;display:flex;align-items:center;gap:12px'>
        <span style='font-size:1.2rem'>🤖</span>
        <div style='font-family:DM Sans;font-size:0.82rem;color:#166534'>
            <b>Random Forest Classifier</b> — F1 Score: 0.861 · ROC-AUC: 0.806 · Trained on 211 pharmacy inventory records
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generate AI Inventory Report →", type="primary", use_container_width=True):
        with st.spinner("Running model inference..."):
            try:
                payload = {
                    "Prix_d_achat_HT": prix_achat,
                    "Prix_de_vente_HT": prix_vente,
                    "Prix_de_vente_TTC": prix_ttc,
                    "Famille": famille,
                    "Designation": designation,
                    "Libelle": designation,
                    "Ville": ville,
                    "Stock_Duration_Days": 30.0
                }
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                res = response.json()

                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                with m1: st.metric("Predicted Status", res["label"])
                with m2: st.metric("AI Confidence",    f"{res['confidence']*100:.1f}%")
                with m3: st.metric("Stock Risk",        "CRITICAL" if res["prediction"] == 1 else "STABLE")

                st.markdown("<br>", unsafe_allow_html=True)

                if res["prediction"] == 1:
                    st.markdown(f"""
                    <div class="alert-low">
                        <div style='font-family:DM Serif Display,serif;font-size:1.3rem;color:#991b1b;margin-bottom:8px'>⚠️ Low Stock Detected</div>
                        <div style='font-family:DM Sans;font-size:0.9rem;color:#7f1d1d;line-height:1.65'>
                            <b>{designation}</b> is at risk of depletion based on forecasted demand patterns for the <b>{famille}</b> category in <b>{ville}</b>.<br>
                            <span style='font-weight:600;margin-top:8px;display:inline-block'>Recommendation: Urgent inventory restock required.</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-high">
                        <div style='font-family:DM Serif Display,serif;font-size:1.3rem;color:#166534;margin-bottom:8px'>✅ Stock Level Optimal</div>
                        <div style='font-family:DM Sans;font-size:0.9rem;color:#14532d;line-height:1.65'>
                            <b>{designation}</b> has sufficient coverage for the projected period.<br>
                            <span style='font-weight:600;margin-top:8px;display:inline-block'>Recommendation: No immediate action required.</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="alert-info">
                    <div style='font-family:DM Sans;font-weight:600;color:#1e40af;margin-bottom:4px'>API Offline</div>
                    <div style='font-family:DM Sans;font-size:0.85rem;color:#1e3a8a'>The ML engine is not reachable. Ensure the FastAPI container is running.<br><code>{str(e)}</code></div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────
def page_dashboard():
    user = st.session_state.username
    info = USER_DASHBOARDS[user]
    section_header("Business Analytics", f"Personalized insights for {info['name']}", "📊")

    url = info["url"]
    if "REPLACE" in url:
        st.markdown("""
        <div class="alert-info">
            <div style='font-family:DM Sans;font-weight:600;color:#1e40af;margin-bottom:4px'>Dashboard pending integration</div>
            <div style='font-family:DM Sans;font-size:0.85rem;color:#1e3a8a'>This role's Power BI dashboard is being provisioned.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#f0fdf4;border:1px solid #dcfce7;border-radius:10px;padding:12px 18px;margin-bottom:16px;font-family:DM Sans;font-size:0.82rem;color:#166534'>
            📌 <b>Tip:</b> You may need to sign in with your Microsoft account to view the embedded Power BI report.
        </div>
        """, unsafe_allow_html=True)
        components.iframe(src=url, height=800, scrolling=True)


# ─────────────────────────────────────────
# PAGE: PROFITABILITY
# ─────────────────────────────────────────
def page_profitability():
    section_header("Profit Analysis", "Simulate pricing and project revenue using AI-driven volume estimation", "💰")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div style='font-family:DM Sans;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:12px'>Pricing Inputs</div>""", unsafe_allow_html=True)
        prix_achat = st.number_input("Unit Purchase Price HT (DT)", value=25.5, step=0.1, key="prof_achat")
        prix_vente = st.number_input("Unit Selling Price HT (DT)",  value=35.0, step=0.1, key="prof_vente")
        prix_ttc   = st.number_input("Unit Selling Price TTC (DT)", value=41.3, step=0.1, key="prof_ttc")
    with col2:
        st.markdown("""<div style='font-family:DM Sans;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:12px'>Product Segment</div>""", unsafe_allow_html=True)
        famille     = st.selectbox("Product Family", ["TOILETTE","COMPRIME","COMP. ALIM.","OPH-ORL","POMMADE","SUPPOSITOIRE","DIVERS","NATURE"], key="prof_famille")
        designation = st.text_input("Designation", value="SHAMPOO", key="prof_desig")
        ville       = st.text_input("City Location", value="Tunis", key="prof_ville")

    if st.button("Calculate Profit Potential →", type="primary", use_container_width=True):
        with st.spinner("Analyzing market potential..."):
            try:
                payload = {
                    "Prix_d_achat_HT": prix_achat,
                    "Prix_de_vente_HT": prix_vente,
                    "Prix_de_vente_TTC": prix_ttc,
                    "Famille": famille,
                    "Designation": designation,
                    "Libelle": designation,
                    "Ville": ville,
                    "Stock_Duration_Days": 30.0
                }
                response = requests.post(f"{API_URL}/predict/profitability", json=payload, timeout=5)

                if response.status_code != 200:
                    st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
                else:
                    res = response.json()
                    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

                    m1, m2, m3, m4 = st.columns(4)
                    with m1: st.metric("Expected Volume",   f"{res['predicted_volume']} units")
                    with m2: st.metric("Margin / Unit",     f"{res['margin_per_unit']} DT")
                    with m3: st.metric("Estimated Profit",  f"{res['predicted_profit']} DT")
                    with m4: st.metric("ROI",               f"{res['roi_percent']}%")

                    st.markdown("<br>", unsafe_allow_html=True)
                    roi    = res['roi_percent']
                    profit = res['predicted_profit']

                    if roi > 40 and profit > 100:
                        tier, color, bg, desc = "🌟 Strategic Asset", "#166534", "#f0fdf4", "High margin and high volume. This product is a primary profit driver — prioritize stocking and promotion."
                    elif roi > 30:
                        tier, color, bg, desc = "✅ Stable Performer", "#1e40af", "#eff6ff", "Healthy margins with steady demand. Essential for operational sustainability."
                    elif profit > 50:
                        tier, color, bg, desc = "📦 Volume Driver", "#92400e", "#fffbeb", "Lower margins but high turnover. Good for cash flow — maintain consistent stock."
                    else:
                        tier, color, bg, desc = "⚠️ Low Priority",   "#374151", "#f9fafb", "Limited profit potential. Consider optimizing pricing or reducing procurement."

                    st.markdown(f"""
                    <div style='background:{bg};border:1px solid {color}30;border-left:6px solid {color};border-radius:16px;padding:28px'>
                        <div style='font-family:DM Serif Display,serif;font-size:1.5rem;color:{color};margin-bottom:10px'>{tier}</div>
                        <div style='font-family:DM Sans;font-size:0.9rem;color:{color};opacity:0.85;line-height:1.65'>{desc}</div>
                        <div style='margin-top:16px;font-family:DM Sans;font-size:0.78rem;color:#9ca3af;font-style:italic'>Based on historical data for {famille} products in {ville}</div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="alert-info">
                    <div style='font-family:DM Sans;font-weight:600;color:#1e40af'>API Offline</div>
                    <div style='font-family:DM Sans;font-size:0.85rem;color:#1e3a8a'>{str(e)}</div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE: FORECASTING
# ─────────────────────────────────────────
def page_forecasting():
    section_header("Time Series Forecasting", "Monthly stock quantity forecasting using Prophet and SARIMA", "⏱")

    col1, col2 = st.columns([1, 2])
    with col1:
        model_type = st.radio("Forecasting Model", ["Prophet", "SARIMA"], horizontal=False)
        st.markdown("""
        <div style='background:#f0fdf4;border:1px solid #dcfce7;border-radius:10px;padding:14px;margin-top:12px;font-family:DM Sans;font-size:0.82rem;color:#166534;line-height:1.65'>
            <b>Prophet</b> — RMSE: 179.31, MAE: 142.77, MAPE: 252.9%<br><br>
            <b>SARIMA</b> — RMSE: 180.35, MAE: 157.81, MAPE: 496.6%<br><br>
            Prophet is the recommended model for this dataset.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        try:
            forecast_model    = model_type.lower()
            history_response  = requests.get(f"{API_URL}/forecast/history", timeout=5)
            forecast_response = requests.get(f"{API_URL}/forecast?model_type={forecast_model}", timeout=5)
            history_data      = history_response.json()
            forecast_data     = forecast_response.json()

            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Model",       forecast_data["model"])
            with m2: st.metric("RMSE",        f"{forecast_data['rmse']:.2f}")
            with m3: st.metric("MAE",         f"{forecast_data['mae']:.2f}")
            with m4: st.metric("Test Points", len(forecast_data['forecast_values']))

            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                if 'train' in history_data:
                    fig.add_trace(go.Scatter(x=history_data['train']['dates'], y=history_data['train']['values'],
                        mode='lines', name='Training Data', line=dict(color='#16a34a', width=2)))
                fig.add_trace(go.Scatter(x=forecast_data['actual_dates'], y=forecast_data['actual_values'],
                    mode='lines+markers', name='Actual', line=dict(color='#052e16', width=2), marker=dict(size=7)))
                fig.add_trace(go.Scatter(x=forecast_data['forecast_dates'], y=forecast_data['forecast_values'],
                    mode='lines+markers', name=f'{model_type} Forecast',
                    line=dict(color='#f59e0b', width=2, dash='dash'), marker=dict(size=7)))
                fig.update_layout(height=380, template='plotly_white', hovermode='x unified',
                    legend=dict(orientation='h', y=-0.15),
                    xaxis_title='Date', yaxis_title='Stock Quantity (Units)')
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.markdown(f"""
            <div class="alert-info">
                <div style='font-family:DM Sans;font-weight:600;color:#1e40af'>Forecasting API Unavailable</div>
                <div style='font-family:DM Sans;font-size:0.85rem;color:#1e3a8a'>{str(e)}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────
if not st.session_state.authenticated:
    if st.session_state.show_login:
        show_login()
    else:
        show_landing()
else:
    show_sidebar()
    page = st.session_state.page
    if   page == 'prediction':    page_prediction()
    elif page == 'dashboard':     page_dashboard()
    elif page == 'profitability': page_profitability()
    elif page == 'forecasting':   page_forecasting()
    else:                         page_prediction()