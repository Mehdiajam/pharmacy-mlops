# ============================================================
# app.py — Multi-User Pharmacy MLOps Dashboard
# ============================================================

import streamlit as st
import requests
import json
import pandas as pd
import streamlit.components.v1 as components

# ─────────────────────────────────────────
# CONFIG & ICONS
# ─────────────────────────────────────────
API_URL = "http://fastapi:8000"

ICONS = {
    "pharmacy": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-pill"><path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/></svg>',
    "dashboard": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-layout-dashboard"><rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/></svg>',
    "prediction": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-sparkles"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>',
    "logout": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-log-out"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" x2="9" y1="12" y2="12"/></svg>',
    "user": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-user"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    "trending": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trending-up"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "box": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-package"><path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.27 6.96 8.73 5.05 8.73-5.05"/><path d="M12 22.08V12"/></svg>',
    "price": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-banknote"><rect width="20" height="12" x="2" y="6" rx="2"/><circle cx="12" cy="12" r="2"/><path d="M6 12h.01"/><path d="M18 12h.01"/></svg>'
}

def get_icon(name, size=24, color="currentColor"):
    return ICONS.get(name, "").replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"').replace('stroke="currentColor"', f'stroke="{color}"')

# Map users to their specific dashboard links
USER_DASHBOARDS = {
    "admin": {
        "password": "admin123",
        "name": "Head Office Admin",
        "url": "https://app.powerbi.com/reportEmbed?reportId=408f00aa-8560-4180-9906-175f960ea7ac&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730"
    },
    "manager": {
        "password": "manager123",
        "name": "Store Manager",
        "url": "REPLACE_WITH_YOUR_SECOND_LINK"
    },
    "pharmacist": {
        "password": "pharmacy123",
        "name": "Lead Pharmacist",
        "url": "REPLACE_WITH_YOUR_THIRD_LINK"
    }
}

st.set_page_config(
    page_title="PharmaStock AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'prediction'

# ─────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@300;400;500;600;700&family=Noto+Sans:wght@300;400;500;700&display=swap');

    :root {
        --primary: #15803D;
        --secondary: #22C55E;
        --accent: #0369A1;
        --background: #F0FDF4;
        --text: #14532D;
        --white: #FFFFFF;
        --shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Noto Sans', sans-serif;
        background-color: var(--background);
        color: var(--text);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Figtree', sans-serif !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--white);
        border-right: 1px solid #E5E7EB;
    }
    
    /* Result Boxes */
    .result-card {
        background: var(--white);
        padding: 24px;
        border-radius: 16px;
        box-shadow: var(--shadow);
        border-left: 6px solid var(--primary);
        margin-bottom: 20px;
    }
    .result-low { border-left-color: #EF4444; background: #FEF2F2; }
    .result-high { border-left-color: #22C55E; background: #F0FDF4; }
    
    /* Form & Input Styling */
    [data-testid="stForm"] {
        background: var(--white);
        padding: 40px;
        border-radius: 20px;
        border: none;
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1);
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stMetric {
        background: var(--white);
        padding: 15px;
        border-radius: 12px;
        box-shadow: var(--shadow);
    }

    /* Custom Icon Wrapper */
    .icon-wrapper {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# AUTHENTICATION FUNCTIONS
# ─────────────────────────────────────────
def show_login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 30px;'>
            <div style='display: inline-block; padding: 20px; background: white; border-radius: 50%; box-shadow: var(--shadow); color: var(--primary);'>
                {get_icon("pharmacy", size=48)}
            </div>
            <h1 style='color: var(--primary); margin-top: 15px; margin-bottom: 0;'>PharmaStock AI</h1>
            <p style='color: var(--text); opacity: 0.8; margin-top: 5px; font-weight: 500;'>Inventory Intelligence Portal</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("<h4 style='text-align: center; margin-bottom: 20px;'>Secure Member Access</h4>", unsafe_allow_html=True)
            user = st.text_input("Username", placeholder="admin, manager, or pharmacist").lower().strip()
            pwd = st.text_input("Password", type="password", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if user in USER_DASHBOARDS and USER_DASHBOARDS[user]["password"] == pwd:
                    st.session_state.authenticated = True
                    st.session_state.username = user
                    st.success("Authentication successful. Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please verify your details.")

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.page = 'prediction'
    st.rerun()

# ─────────────────────────────────────────
# MAIN APP LOGIC
# ─────────────────────────────────────────
if not st.session_state.authenticated:
    show_login()
else:
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown(f"""
        <div style='display: flex; align-items: center; margin-bottom: 20px;'>
            <div style='color: var(--primary); margin-right: 12px;'>{get_icon("user", size=32)}</div>
            <div>
                <div style='font-weight: 700; color: var(--text); font-size: 1.1rem;'>{USER_DASHBOARDS[st.session_state.username]['name']}</div>
                <div style='font-size: 0.8rem; color: var(--text); opacity: 0.6;'>{st.session_state.username.upper()} SESSION</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation
        st.markdown("<p style='font-weight: 600; font-size: 0.75rem; color: var(--text); opacity: 0.5; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em;'>Main Menu</p>", unsafe_allow_html=True)
        
        if st.button("Stock Prediction", use_container_width=True, 
                     type="primary" if st.session_state.page == 'prediction' else "secondary"):
            st.session_state.page = 'prediction'
            st.rerun()
            
        if st.button("Business Analytics", use_container_width=True, 
                     type="primary" if st.session_state.page == 'dashboard' else "secondary"):
            st.session_state.page = 'dashboard'
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            logout()
        
        st.markdown("---")
        # System Health
        try:
            health = requests.get(f"{API_URL}/health", timeout=1).json()
            if health["model_loaded"]:
                st.markdown(f"""
                <div style='display: flex; align-items: center; padding: 10px; background: #F0FDF4; border-radius: 8px; border: 1px solid #DCFCE7;'>
                    <div style='width: 8px; height: 8px; background: #22C55E; border-radius: 50%; margin-right: 10px; box-shadow: 0 0 8px #22C55E;'></div>
                    <div style='font-size: 0.8rem; font-weight: 600; color: #15803D;'>ML ENGINE ONLINE</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='display: flex; align-items: center; padding: 10px; background: #FFFBEB; border-radius: 8px; border: 1px solid #FEF3C7;'>
                    <div style='width: 8px; height: 8px; background: #F59E0B; border-radius: 50%; margin-right: 10px;'></div>
                    <div style='font-size: 0.8rem; font-weight: 600; color: #92400E;'>ENGINE LOADING...</div>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown(f"""
            <div style='display: flex; align-items: center; padding: 10px; background: #FEF2F2; border-radius: 8px; border: 1px solid #FEE2E2;'>
                <div style='width: 8px; height: 8px; background: #EF4444; border-radius: 50%; margin-right: 10px;'></div>
                <div style='font-size: 0.8rem; font-weight: 600; color: #991B1B;'>API OFFLINE</div>
            </div>
            """, unsafe_allow_html=True)

    # --- PAGE: DASHBOARD ---
    if st.session_state.page == 'dashboard':
        st.markdown(f"<h1>{get_icon('dashboard', size=32)} Business Analytics</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; opacity: 0.8; margin-bottom: 30px;'>Personalized business intelligence insights for <b>{USER_DASHBOARDS[st.session_state.username]['name']}</b>.</p>", unsafe_allow_html=True)
        
        url = USER_DASHBOARDS[st.session_state.username]["url"]
        
        if "REPLACE" in url:
            st.info(f"The specialized dashboard for the **{st.session_state.username}** role is currently being provisioned.")
            st.image("https://illustrations.popsy.co/green/presentation.svg", width=500)
            st.markdown(f"""
            <div style='background: white; padding: 30px; border-radius: 16px; box-shadow: var(--shadow); border: 1px dashed #D1D5DB; text-align: center;'>
                <h4 style='color: #6B7280 !important;'>Integration Pending</h4>
                <p style='color: #9CA3AF;'>Role: {st.session_state.username.upper()}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            components.iframe(src=url, height=800, scrolling=True)

    # --- PAGE: PREDICTION ---
    else:
        st.markdown(f"<h1>{get_icon('prediction', size=32)} Smart Stock Prediction</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.1rem; opacity: 0.8; margin-bottom: 30px;'>Forecast inventory needs using our trained Random Forest model.</p>", unsafe_allow_html=True)
        
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                    <div style='color: var(--accent); margin-right: 10px;'>{get_icon("price", size=24)}</div>
                    <h4 style='margin: 0;'>Pricing Structure</h4>
                </div>
                """, unsafe_allow_html=True)
                prix_achat = st.number_input("Purchase Price HT (DT)", value=25.5, step=0.1)
                prix_vente = st.number_input("Selling Price HT (DT)", value=35.0, step=0.1)
                prix_ttc   = st.number_input("Selling Price TTC (DT)", value=41.3, step=0.1)
            with col2:
                st.markdown(f"""
                <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                    <div style='color: var(--accent); margin-right: 10px;'>{get_icon("box", size=24)}</div>
                    <h4 style='margin: 0;'>Product Attributes</h4>
                </div>
                """, unsafe_allow_html=True)
                famille = st.selectbox("Product Family", ["TOILETTE", "COMPRIME", "COMP. ALIM.", "OPH-ORL", "POMMADE", "SUPPOSITOIRE", "DIVERS", "NATURE"])
                designation = st.text_input("Designation", value="SHAMPOO")
                ville = st.text_input("City Location", value="Tunis")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate AI Inventory Report", type="primary", use_container_width=True):
            with st.spinner("Processing Model Inference..."):
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
                    
                    st.markdown("<br><hr>", unsafe_allow_html=True)
                    st.markdown(f"<h3>{get_icon('trending', size=28)} Prediction Summary</h3>", unsafe_allow_html=True)
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Predicted Status", res["label"])
                    with m2:
                        st.metric("AI Confidence", f"{res['confidence']*100:.1f}%")
                    with m3:
                        risk_level = "CRITICAL" if res["prediction"] == 1 else "STABLE"
                        st.metric("Stock Risk", risk_level)

                    if res["prediction"] == 1:
                        st.markdown(f"""
                        <div class="result-card result-low">
                            <div style='display: flex; align-items: flex-start;'>
                                <div style='color: #EF4444; margin-right: 15px; margin-top: 2px;'>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
                                </div>
                                <div>
                                    <h3 style='margin:0; color: #991B1B !important;'>LOW STOCK DETECTED</h3>
                                    <p style='margin: 10px 0; color: #7F1D1D;'>Product <b>{designation}</b> is currently at risk of depletion based on forecasted demand patterns.</p>
                                    <p style='margin:0; font-weight: 600; color: #991B1B;'>REFILL RECOMMENDATION: Urgent inventory restock required.</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card result-high">
                            <div style='display: flex; align-items: flex-start;'>
                                <div style='color: #22C55E; margin-right: 15px; margin-top: 2px;'>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                                </div>
                                <div>
                                    <h3 style='margin:0; color: #166534 !important;'>STOCK LEVEL OPTIMAL</h3>
                                    <p style='margin: 10px 0; color: #14532D;'>Product <b>{designation}</b> has sufficient coverage for the projected period.</p>
                                    <p style='margin:0; font-weight: 600; color: #166534;'>REFILL RECOMMENDATION: No immediate action required.</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Inference Engine Error: {str(e)}")
