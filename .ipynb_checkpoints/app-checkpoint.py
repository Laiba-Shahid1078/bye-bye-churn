import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# ==========================================
# 1. PAGE CONFIGURATION & PREMIUM CSS
# ==========================================
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Custom CSS matching the exact Cyber/Teal Glassmorphic Theme from the Reference Image
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Global Reset & Typography */
* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* App Background: Deep Obsidian / Cyber-Navy Radial Gradient */
.stApp {
    background: radial-gradient(circle at 50% 12%, #0d2235 0%, #060e18 55%, #03070d 100%) !important;
    color: #e2e8f0 !important;
}

/* Hide Default Streamlit Chrome */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 96% !important;
}

/* Sidebar Deep Glassmorphic Theming */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071320 0%, #030810 100%) !important;
    border-right: 1px solid rgba(0, 242, 254, 0.12) !important;
}

[data-testid="stSidebar"] hr {
    border-color: #ffffff !important;
    opacity: 0.35 !important;
    margin: 1.2rem 0 !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    transition: color 0.2s ease;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    color: #00f2fe !important;
}

/* Sidebar Caption & Divider in White */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #ffffff !important;
    opacity: 0.95 !important;
}

/* Premium KPI Cards (Glassmorphism & Neon Glow from Reference) */
.kpi-card {
    background: linear-gradient(135deg, rgba(14, 28, 44, 0.75) 0%, rgba(9, 18, 30, 0.85) 100%);
    border: 1px solid rgba(0, 242, 254, 0.14);
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(12px);
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 242, 254, 0.45), transparent);
}

.kpi-card:hover {
    transform: translateY(-4px);
    border-color: #00f2fe;
    box-shadow: 0 10px 30px rgba(0, 242, 254, 0.2);
}

.kpi-title {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
    font-weight: 600;
}

.kpi-value {
    font-size: 34px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
    line-height: 1.1;
}

.kpi-alert {
    font-size: 11px;
    color: #00f5a0;
    background: rgba(0, 245, 160, 0.12);
    border: 1px solid rgba(0, 245, 160, 0.25);
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
    font-weight: 700;
    letter-spacing: 0.3px;
}

.kpi-danger {
    color: #ff4d6d !important;
    background: rgba(255, 77, 109, 0.12) !important;
    border-color: rgba(255, 77, 109, 0.25) !important;
}

/* DataFrame Customization */
[data-testid="stDataFrame"] {
    background-color: rgba(10, 22, 35, 0.7) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 242, 254, 0.15) !important;
    backdrop-filter: blur(10px);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: 1px solid rgba(0, 242, 254, 0.35) !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(0, 180, 216, 0.2) !important;
    transition: all 0.25s ease !important;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #0096c7 0%, #00f2fe 100%) !important;
    border-color: #00f2fe !important;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.45) !important;
    color: #ffffff !important;
}

/* Selectbox, Radio, and Slider Theming */
div[data-baseweb="select"] > div {
    background-color: #0b1a29 !important;
    border-color: #16364d !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: #00f2fe !important;
}

.stSlider > div [data-baseweb="slider"] {
    color: #00f2fe !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 2. DATA & MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("bye_bye_churn_pipeline.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/risk_scores.csv")

try:
    model = load_model()
    risk_df = load_data()
except Exception as e:
    st.error(f"⚠️ Initialization Error: {e}. Model ya Data load nahi ho saka.")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px 0;'>
    <h2 style='color: #00f2fe; font-weight: 800; letter-spacing: 1px; margin: 0; text-shadow: 0 0 15px rgba(0,242,254,0.4);'>
        🌐 SOC ANALYTICS
    </h2>
    <div style='font-size: 11px; color: #64748b; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 4px;'>
        Threat Intelligence Radar
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "📊 Executive Dashboard", 
    "🎯 Risk Segmentation", 
    "🧪 What-If Simulator", 
    "🧠 SHAP Analytics"
])
st.sidebar.markdown("<hr style='border-color: #ffffff; opacity: 0.35; margin: 1.2rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #ffffff; font-size: 13px; font-weight: 500; margin: 0;'>Model Engine: Tuned XGBoost</p>", unsafe_allow_html=True)

# ==========================================
# 4. MODULES IMPLEMENTATION
# ==========================================

# ------------------------------------------
# MODULE 1: EXECUTIVE DASHBOARD
# ------------------------------------------
if page == "📊 Executive Dashboard":
    st.markdown("<h1 style='color: #f8fafc; font-weight: 700; margin-bottom: 4px;'>Real-Time Intelligence & Risk Monitoring</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 15px;'>Live telemetry of financial exposure and customer attrition metrics.</p><br>", unsafe_allow_html=True)
    
    # KPIs Calculation
    total_customers = len(risk_df)
    high_risk_count = len(risk_df[risk_df["risk_band"] == "High"])
    total_rev_risk = risk_df["annual_revenue_at_risk"].sum()
    high_rev_risk = risk_df[risk_df["risk_band"] == "High"]["annual_revenue_at_risk"].sum()
    
    # Custom HTML Metric Cards matching Reference Card Aesthetics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Sources</div>
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-alert">Protected</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">High-Risk Incidents</div>
            <div class="kpi-value">{high_risk_count:,}</div>
            <div class="kpi-alert kpi-danger">Critical Threat</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Global Threat Radar</div>
            <div class="kpi-value">${total_rev_risk / 1e6:.2f}M</div>
            <div class="kpi-alert">Annual Forecast</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Immediate Exposure</div>
            <div class="kpi-value">${high_rev_risk / 1e6:.2f}M</div>
            <div class="kpi-alert kpi-danger">Action Required</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Dark Mode Plotly Charts (Refined to the exact reference neon teal / coral / amber palette)
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("<h4 style='color: #f8fafc; font-weight: 600;'>Threat Activity Distribution</h4>", unsafe_allow_html=True)
        pie_fig = px.pie(
            risk_df, names='risk_band', 
            color='risk_band',
            color_discrete_map={'High':'#ff4d6d', 'Medium':'#fbbf24', 'Low':'#00f5a0'},
            hole=0.62, template="plotly_dark"
        )
        pie_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True,
            font=dict(family="Plus Jakarta Sans", color="#94a3b8")
        )
        st.plotly_chart(pie_fig, use_container_width=True)
        
    with col_chart2:
        st.markdown("<h4 style='color: #f8fafc; font-weight: 600;'>Financial Attack Volume</h4>", unsafe_allow_html=True)
        bar_df = risk_df.groupby("risk_band")["annual_revenue_at_risk"].sum().reset_index()
        bar_fig = px.bar(
            bar_df, x='risk_band', y='annual_revenue_at_risk', 
            color='risk_band',
            color_discrete_map={'High':'#ff4d6d', 'Medium':'#fbbf24', 'Low':'#00f5a0'},
            text_auto='.2s', template="plotly_dark"
        )
        bar_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis_title="Risk Level",
            yaxis_title="Revenue at Risk ($)",
            font=dict(family="Plus Jakarta Sans", color="#94a3b8"),
            xaxis=dict(gridcolor="#16384c"),
            yaxis=dict(gridcolor="#16384c")
        )
        st.plotly_chart(bar_fig, use_container_width=True)

# ------------------------------------------
# MODULE 2: RISK SEGMENTATION
# ------------------------------------------
elif page == "🎯 Risk Segmentation":
    st.markdown("<h1 style='color: #f8fafc; font-weight: 700;'>AI Threat Detection</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_band = st.selectbox("Isolate Risk Level", ["High", "Medium", "Low", "All"])
    
    filtered_df = risk_df[risk_df["risk_band"] == selected_band] if selected_band != "All" else risk_df
    st.markdown(f"<p style='color: #00f2fe; font-weight: 600; font-size: 15px;'>Monitoring {len(filtered_df):,} Targets</p>", unsafe_allow_html=True)
    
    display_cols = ["customerID", "risk_band", "churn_probability", "MonthlyCharges", "annual_revenue_at_risk", "top_3_churn_reasons"]
    st.dataframe(filtered_df[display_cols].sort_values(by="churn_probability", ascending=False), use_container_width=True, hide_index=True)

# ------------------------------------------
# MODULE 3: WHAT-IF SIMULATOR
# ------------------------------------------
elif page == "🧪 What-If Simulator":
    st.markdown("<h1 style='color: #f8fafc; font-weight: 700;'>Threat Mitigation Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 15px;'>Deploy countermeasures and simulate risk reduction live.</p><br>", unsafe_allow_html=True)
    
    high_risk_customers = risk_df[risk_df["risk_band"] == "High"]["customerID"].tolist()
    selected_customer = st.selectbox("Select Target Anomaly (High Risk)", high_risk_customers)
    
    if selected_customer:
        cust_data = risk_df[risk_df["customerID"] == selected_customer].iloc[0].to_dict()
        current_prob = cust_data["churn_probability"]
        
        st.markdown(f"<h3 style='color: #00f2fe; font-weight: 600;'>Target ID: {selected_customer} | Baseline Threat: {current_prob:.1%}</h3><hr style='border-color: rgba(0, 242, 254, 0.2);'>", unsafe_allow_html=True)
        
        col_form, col_gauge = st.columns([1, 1])
        
        with col_form:
            new_contract = st.selectbox("Modify Contract Strategy", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(cust_data["Contract"]))
            new_internet = st.selectbox("Adjust Infrastructure", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(cust_data["InternetService"]))
            new_tech_support = st.radio("Deploy Security Support?", ["Yes", "No"], index=["Yes", "No"].index(cust_data["TechSupport"]), horizontal=True)
            new_monthly_charge = st.slider("Optimize Cost Parameters ($)", 15.0, 120.0, float(cust_data["MonthlyCharges"]))
            simulate = st.button("Execute Simulation", type="primary", use_container_width=True)
            
        with col_gauge:
            sim_df = pd.DataFrame([cust_data])
            # Passed values as lists to prevent ValueError
            sim_df.update(pd.DataFrame({
                "Contract": [new_contract], 
                "InternetService": [new_internet], 
                "TechSupport": [new_tech_support], 
                "MonthlyCharges": [new_monthly_charge]
            }))
            
            cols_to_drop = ["churn_probability", "risk_band", "expected_monthly_loss", "annual_revenue_at_risk", "top_3_churn_reasons", "actual_churn", "data_split"]
            sim_df_clean = sim_df.drop(columns=[c for c in cols_to_drop if c in sim_df.columns])
            
            if simulate:
                new_prob = model.predict_proba(sim_df_clean)[0][1]
                
                gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = new_prob * 100,
                    delta = {'reference': current_prob * 100, 'position': "top", 'increasing': {'color': "#ff4d6d"}, 'decreasing': {'color': "#00f5a0"}},
                    number = {'suffix': "%", 'font': {'color': '#f8fafc', 'size': 38}},
                    title = {'text': "Simulated Threat Level", 'font': {'color': '#94a3b8', 'family': 'Plus Jakarta Sans'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': "#16384c"},
                        'bar': {'color': "#00f2fe", 'thickness': 0.3},
                        'bgcolor': "#0a1928",
                        'borderwidth': 1,
                        'bordercolor': "#16384c",
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(0, 245, 160, 0.2)"},
                            {'range': [30, 60], 'color': "rgba(251, 191, 36, 0.2)"},
                            {'range': [60, 100], 'color': "rgba(255, 77, 109, 0.2)"}
                        ]
                    }
                ))
                gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
                st.plotly_chart(gauge, use_container_width=True)

# ------------------------------------------
# MODULE 4: SHAP ANALYTICS
# ------------------------------------------
elif page == "🧠 SHAP Analytics":
    st.markdown("<h1 style='color: #f8fafc; font-weight: 700;'>AI Confidence & Threat Attribution</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 15px;'>Global feature impact analysis detecting anomaly sources.</p><br>", unsafe_allow_html=True)
    
    try:
        st.image("reports/figures/shap_summary.png", use_container_width=True)
    except Exception:
        st.error("SHAP plot missing. Please ensure Phase 4 was executed.")