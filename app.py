"""
Bye Bye Churn: Retention Intelligence Dashboard
------------------------------------------------
Run:   streamlit run Home.py

Files needed (project root or relative to script):
  - bye_bye_churn_pipeline.joblib            (trained pipeline with predict_proba)
  - data/processed/risk_scores.csv           (customerID, churn_probability, risk_band,
                                              MonthlyCharges, annual_revenue_at_risk, ...)
  - reports/figures/shap_summary.png         (optional, for SHAP page)

Optional columns that enrich dashboard capabilities:
  actual_churn, data_split, top_3_churn_reasons, tenure, Contract, InternetService,
  PaymentMethod, TechSupport, expected_monthly_loss
"""
import inspect
import os
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# 0. PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Bye Bye Churn | Retention Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILES = ["bye_bye_churn_pipeline.joblib", "models/bye_bye_churn_pipeline.joblib"]
DATA_FILES = ["data/processed/risk_scores.csv", "risk_scores.csv"]
SHAP_FILES = ["reports/figures/shap_summary.png", "shap_summary.png"]

C = dict(
    bg="#060E18", panel="#0B1A29", line="#16364D", text="#E2E8F0", mute="#94A3B8",
    accent="#00F2FE", high="#FF4D6D", med="#FBBF24", low="#00F5A0",
)
BAND_ORDER = ["Low", "Medium", "High"]
BAND_COLORS = {"Low": C["low"], "Medium": C["med"], "High": C["high"]}
RISK_SCALE = [[0.0, C["low"]], [0.5, C["med"]], [1.0, C["high"]]]
TENURE_LABELS = ["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]

# Operational columns excluded from model inputs (dashboard-engineered features)
DASH_COLS = [
    "churn_probability", "risk_band", "expected_monthly_loss", "annual_revenue_at_risk",
    "top_3_churn_reasons", "actual_churn", "data_split", "tenure_band", "churn_pct",
]
PROFILE_SKIP = set(DASH_COLS) | {"customerID"}

# =====================================================================
# 1. STYLE
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: radial-gradient(circle at 50% 0%, #0d2235 0%, #060e18 55%, #03070d 100%) !important;
    color: #e2e8f0;
}
button, input, textarea, select { font-family: inherit; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 96%; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071320 0%, #030810 100%) !important;
    border-right: 1px solid rgba(0, 242, 254, 0.12);
}
[data-testid="stSidebar"] .stRadio label { color: #94a3b8; font-weight: 500; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover { color: #00f2fe; }
.brand { padding: 6px 4px 2px 4px; }
.brand b { color: #00f2fe; font-size: 21px; font-weight: 800; letter-spacing: .2px; }
.brand span { display: block; color: #64748b; font-size: 12px; margin-top: 2px; }
.side-title { color: #f8fafc; font-weight: 700; font-size: 13px; margin: 6px 0 2px 0; }
.side-foot { color: #94a3b8; font-size: 12px; line-height: 1.5; }
.side-foot b { color: #f8fafc; font-weight: 600; }

/* ---------- Page header ---------- */
.page-head h1 { color: #f8fafc; font-weight: 800; font-size: 30px; margin: 0 0 2px 0; padding: 0; letter-spacing: -.3px; }
.page-head p { color: #94a3b8; font-size: 14.5px; margin: 0 0 18px 0; }

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, rgba(14,28,44,.85) 0%, rgba(9,18,30,.92) 100%);
    border: 1px solid rgba(0,242,254,.16); border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
    box-shadow: 0 10px 40px rgba(0,0,0,.45);
}
.hero-num { font-size: 54px; font-weight: 800; color: #f8fafc; letter-spacing: -1.5px; line-height: 1; }
.hero-sub { color: #94a3b8; font-size: 15px; margin: 10px 0 18px 0; max-width: 760px; line-height: 1.5; }
.hero-sub b { color: #f8fafc; font-weight: 600; }
.hero-bar { display: flex; height: 12px; border-radius: 99px; overflow: hidden; background: #0a1928; gap: 2px; }
.hero-bar div { height: 100%; }
.hero-leg { display: flex; flex-wrap: wrap; gap: 22px; margin-top: 12px; color: #94a3b8; font-size: 13px; }
.hero-leg i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }

/* ---------- KPI cards ---------- */
.kpi-card {
    background: linear-gradient(135deg, rgba(14,28,44,.75) 0%, rgba(9,18,30,.85) 100%);
    border: 1px solid rgba(0,242,254,.13); border-radius: 14px; padding: 18px 18px 16px 18px;
    box-shadow: 0 8px 28px rgba(0,0,0,.35); height: 100%;
}
.kpi-title { font-size: 13px; color: #94a3b8; margin-bottom: 6px; font-weight: 500; }
.kpi-value { font-size: 30px; font-weight: 800; color: #f8fafc; letter-spacing: -.5px; line-height: 1.1; }
.kpi-sub { font-size: 12.5px; margin-top: 8px; color: #94a3b8; }
.kpi-sub.good { color: #00f5a0; } .kpi-sub.bad { color: #ff4d6d; } .kpi-sub.warn { color: #fbbf24; }

/* ---------- Cards / containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(10,22,35,.55); border-color: rgba(0,242,254,.12) !important; border-radius: 14px;
}
.card-t { color: #f8fafc; font-weight: 700; font-size: 16px; margin: 0 0 2px 0; }
.card-s { color: #94a3b8; font-size: 13px; margin: 0 0 8px 0; }
.note { color: #94a3b8; font-size: 13px; border-left: 3px solid #00f2fe; padding: 4px 0 4px 12px; margin: 6px 0 14px 0; }
.chip { display: inline-block; padding: 5px 12px; margin: 0 8px 8px 0; border-radius: 99px; font-size: 12.5px;
        background: rgba(255,77,109,.10); border: 1px solid rgba(255,77,109,.30); color: #ffb3c1; }
.kv { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px 18px; }
.kv div span { display: block; color: #64748b; font-size: 12px; }
.kv div b { color: #f1f5f9; font-weight: 600; font-size: 14px; }

/* ---------- Widgets ---------- */
[data-testid="stMetric"] { background: rgba(10,22,35,.6); border: 1px solid rgba(0,242,254,.12); border-radius: 12px; padding: 12px 16px; }
[data-testid="stMetricLabel"] { color: #94a3b8; }
[data-testid="stDataFrame"] { border: 1px solid rgba(0,242,254,.15); border-radius: 12px; }
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%); color: #fff; border-radius: 8px;
    border: 1px solid rgba(0,242,254,.35); font-weight: 600; transition: all .2s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(135deg, #0096c7 0%, #00f2fe 100%); border-color: #00f2fe; color: #fff;
}
div[data-baseweb="select"] > div { background-color: #0b1a29; border-color: #16364d; border-radius: 8px; }
div[data-baseweb="select"] > div:hover { border-color: #00f2fe; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: rgba(10,22,35,.6); border-radius: 8px 8px 0 0; padding: 8px 16px; color: #94a3b8; }
.stTabs [aria-selected="true"] { color: #00f2fe !important; }
:focus-visible { outline: 2px solid #00f2fe !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =====================================================================
# 2. HELPERS
# =====================================================================
def clean(s: str) -> str:
    """Flatten HTML to a single line to prevent Markdown indentation issues."""
    return re.sub(r"\n\s*", "", s)


def money(x) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.1f}K"
    return f"${x:,.0f}"


def pct(x, d=1) -> str:
    return f"{float(x) * 100:.{d}f}%"


def header(title, subtitle):
    st.markdown(clean(f'<div class="page-head"><h1>{title}</h1><p>{subtitle}</p></div>'), unsafe_allow_html=True)


def card_title(title, sub=""):
    s = f'<div class="card-s">{sub}</div>' if sub else ""
    st.markdown(clean(f'<div class="card-t">{title}</div>{s}'), unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def kpi(col, title, value, sub="", tone="neutral"):
    col.markdown(
        clean(f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
              f'<div class="kpi-value">{value}</div><div class="kpi-sub {tone}">{sub}</div></div>'),
        unsafe_allow_html=True,
    )


def stretch(fn):
    """Compatible with width='stretch' on newer Streamlit and use_container_width=True on older versions."""
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


def chart(fig, key=None):
    st.plotly_chart(fig, key=key, **stretch(st.plotly_chart))


def table(df, **kw):
    st.dataframe(df, hide_index=True, **stretch(st.dataframe), **kw)


def show_image(path):
    st.image(path, **stretch(st.image))


def style(fig, height=340, legend=True):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(t=10, b=10, l=10, r=10), showlegend=legend,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=C["mute"], size=12),
        legend=dict(orientation="h", y=-0.18, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=C["line"], zerolinecolor=C["line"])
    fig.update_yaxes(gridcolor=C["line"], zerolinecolor=C["line"])
    return fig


def find_file(candidates):
    here = os.path.dirname(os.path.abspath(__file__))
    for root in [os.getcwd(), here, os.path.dirname(here)]:
        for c in candidates:
            p = os.path.join(root, c)
            if os.path.exists(p):
                return p
    return None


def parse_reasons(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return [x.strip() for x in re.split(r"[;,|]", str(v)) if x.strip()]


def to_binary(s):
    if s.dtype == bool:
        return s.astype(int)
    m = {"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "true": 1, "false": 0}
    return s.astype(str).str.strip().str.lower().map(m)


def gauge(value, ref=None, title="Churn Risk", lo=0.3, hi=0.6):
    kw = {}
    if ref is not None:
        kw["delta"] = {"reference": ref * 100, "position": "top", "valueformat": ".1f",
                       "increasing": {"color": C["high"]}, "decreasing": {"color": C["low"]}}
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if ref is not None else "gauge+number",
        value=value * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"color": "#f8fafc", "size": 40}},
        title={"text": title, "font": {"color": C["mute"], "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": C["line"]},
            "bar": {"color": C["accent"], "thickness": 0.28},
            "bgcolor": "#0a1928", "borderwidth": 1, "bordercolor": C["line"],
            "steps": [
                {"range": [0, lo * 100], "color": "rgba(0,245,160,.20)"},
                {"range": [lo * 100, hi * 100], "color": "rgba(251,191,36,.20)"},
                {"range": [hi * 100, 100], "color": "rgba(255,77,109,.20)"},
            ],
        },
        **kw,
    ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300,
                      margin=dict(t=60, b=10, l=30, r=30))
    return fig


# =====================================================================
# 3. DATA AND MODEL
# =====================================================================
@st.cache_resource(show_spinner="Loading model pipeline...")
def load_model(path):
    return joblib.load(path)


@st.cache_data(show_spinner="Loading customer risk data...")
def load_data(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def prepare(raw):
    d = raw.copy()
    for c in ["churn_probability", "MonthlyCharges", "tenure", "TotalCharges",
              "annual_revenue_at_risk", "expected_monthly_loss"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "expected_monthly_loss" not in d.columns and {"churn_probability", "MonthlyCharges"} <= set(d.columns):
        d["expected_monthly_loss"] = d["churn_probability"] * d["MonthlyCharges"]
    if "annual_revenue_at_risk" not in d.columns and "expected_monthly_loss" in d.columns:
        d["annual_revenue_at_risk"] = d["expected_monthly_loss"] * 12
    if "risk_band" not in d.columns:
        d["risk_band"] = pd.cut(d["churn_probability"], [-0.01, 0.3, 0.6, 1.0], labels=BAND_ORDER)
    d["risk_band"] = pd.Categorical(d["risk_band"].astype(str), categories=BAND_ORDER, ordered=True)
    if "tenure" in d.columns:
        d["tenure_band"] = pd.cut(d["tenure"], [-1, 12, 24, 48, 10_000], labels=TENURE_LABELS).astype(str)
    return d


def band_cutoffs(df):
    lo = df.loc[df["risk_band"] == "Low", "churn_probability"].max()
    hi = df.loc[df["risk_band"] == "High", "churn_probability"].min()
    return (float(lo) if pd.notna(lo) else 0.3, float(hi) if pd.notna(hi) else 0.6)


def model_label(m):
    names = {"LogisticRegression": "Logistic Regression", "XGBClassifier": "XGBoost",
             "RandomForestClassifier": "Random Forest", "LGBMClassifier": "LightGBM",
             "CatBoostClassifier": "CatBoost"}
    try:
        last = m.steps[-1][1] if hasattr(m, "steps") else m
        prefix = ""
        if type(last).__name__ == "CalibratedClassifierCV":
            prefix = "Calibrated "
            last = getattr(last, "estimator", None) or getattr(last, "base_estimator", last)
            last = last.steps[-1][1] if hasattr(last, "steps") else last
        n = type(last).__name__
        return prefix + names.get(n, n)
    except Exception:
        return type(m).__name__


mp, dp = find_file(MODEL_FILES), find_file(DATA_FILES)
try:
    if mp is None or dp is None:
        missing = [f for f, p in [(MODEL_FILES[0], mp), (DATA_FILES[0], dp)] if p is None]
        raise FileNotFoundError("Missing file(s): " + ", ".join(missing))
    model = load_model(mp)
    DF = prepare(load_data(dp))
except Exception as e:
    st.error(f"Initialization Error: {e}. Model or dataset could not be loaded. "
             "Please check file paths and ensure the application is executed from the project root.")
    st.stop()

LO_CUT, HI_CUT = band_cutoffs(DF)


# ---------------------------------------------------------------- model inputs
def prepare_X(row_df):
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return row_df.reindex(columns=list(names))
    return row_df.drop(columns=[c for c in DASH_COLS if c in row_df.columns])


def predict_prob(base, changes):
    d = {**base, **changes}
    return float(model.predict_proba(prepare_X(pd.DataFrame([d])))[0, 1])


# =====================================================================
# 4. SIDEBAR: BRAND, NAVIGATION, GLOBAL FILTERS
# =====================================================================
PAGES = {
    "📊 Executive Overview": "overview",
    "🎯 Risk Explorer": "explorer",
    "👤 Customer 360": "c360",
    "🧪 What-If Simulator": "whatif",
    "💰 Retention Planner": "planner",
    "📈 Model Health": "health",
    "🧠 SHAP Analytics": "shap",
}

st.sidebar.markdown(
    clean('<div class="brand"><b>📉 Bye Bye Churn</b><span>Retention Intelligence</span></div>'),
    unsafe_allow_html=True,
)
page = st.sidebar.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.markdown("---")


def reset_filters():
    for k, v in st.session_state.get("_defaults", {}).items():
        st.session_state[k] = v


def sidebar_filters(df):
    st.sidebar.markdown('<div class="side-title">Global Filters</div>', unsafe_allow_html=True)
    defaults, mask = {}, pd.Series(True, index=df.index)

    def multi(col, label, options):
        key = f"f_{col}"
        defaults[key] = options
        if key not in st.session_state:
            st.session_state[key] = options
        return st.sidebar.multiselect(label, options, key=key)

    band_sel = multi("risk_band", "Risk Band", [b for b in BAND_ORDER if b in set(df["risk_band"].astype(str))])
    mask &= df["risk_band"].astype(str).isin(band_sel)
    for col, label in [("Contract", "Contract"), ("InternetService", "Internet Service"),
                       ("PaymentMethod", "Payment Method")]:
        if col in df.columns:
            opts = sorted(df[col].dropna().astype(str).unique())
            sel = multi(col, label, opts)
            mask &= df[col].astype(str).isin(sel)

    if "tenure" in df.columns and df["tenure"].notna().any():
        lo, hi = int(df["tenure"].min()), int(df["tenure"].max())
        defaults["f_tenure"] = (lo, hi)
        st.session_state.setdefault("f_tenure", (lo, hi))
        r = st.sidebar.slider("Tenure (Months)", lo, hi, key="f_tenure")
        mask &= df["tenure"].between(r[0], r[1])
    if "MonthlyCharges" in df.columns and df["MonthlyCharges"].notna().any():
        lo, hi = float(np.floor(df["MonthlyCharges"].min())), float(np.ceil(df["MonthlyCharges"].max()))
        defaults["f_charges"] = (lo, hi)
        st.session_state.setdefault("f_charges", (lo, hi))
        r = st.sidebar.slider("Monthly Charges ($)", lo, hi, key="f_charges")
        mask &= df["MonthlyCharges"].between(r[0], r[1])

    st.session_state["_defaults"] = defaults
    st.sidebar.button("Reset Filters", on_click=reset_filters, **stretch(st.button))
    return df[mask]


FDF = sidebar_filters(DF)
st.sidebar.markdown("---")
st.sidebar.markdown(
    clean(f'<div class="side-foot"><b>Model:</b> {model_label(model)}<br>'
          f'<b>Showing:</b> {len(FDF):,} of {len(DF):,} customers</div>'),
    unsafe_allow_html=True,
)

if FDF.empty:
    st.warning("No customers match the current filter criteria. Click 'Reset Filters' in the sidebar to restore data.")
    st.stop()


# =====================================================================
# 5. PAGES
# =====================================================================
def seg_table(df, dim):
    g = (df.assign(_d=df[dim].astype(str)).groupby("_d", observed=True)
         .agg(customers=("churn_probability", "size"),
              avg_risk=("churn_probability", "mean"),
              revenue=("annual_revenue_at_risk", "sum"))
         .reset_index().rename(columns={"_d": dim}))
    return g.sort_values("avg_risk")


# ------------------------------------------------------------- OVERVIEW
def page_overview(df):
    header("Retention Overview", "Identify at-risk customers, quantify revenue exposure, and prioritize intervention cohorts.")

    total = float(df["annual_revenue_at_risk"].sum())
    by = df.groupby("risk_band", observed=False)["annual_revenue_at_risk"].sum().reindex(BAND_ORDER).fillna(0.0)
    n_high = int((df["risk_band"] == "High").sum())
    share_high = by["High"] / total if total > 0 else 0
    segs = "".join(
        f'<div style="width:{(by[b] / total * 100 if total > 0 else 0):.2f}%;background:{BAND_COLORS[b]}"></div>'
        for b in ["High", "Medium", "Low"])
    leg = "".join(f'<span><i style="background:{BAND_COLORS[b]}"></i>{b}: {money(by[b])}</span>'
                  for b in ["High", "Medium", "Low"])
    st.markdown(clean(
        f'<div class="hero"><div class="hero-num">{money(total)}</div>'
        f'<div class="hero-sub">expected annual revenue at risk across <b>{len(df):,} customers</b>. '
        f'<b>{share_high:.0%}</b> is concentrated among <b>{n_high:,} high-risk customers</b>, '
        f'making them the primary candidates for targeted retention campaigns.</div>'
        f'<div class="hero-bar">{segs}</div><div class="hero-leg">{leg}</div></div>'), unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Customers in View", f"{len(df):,}", f"{len(df) / len(DF):.0%} of total cohort")
    kpi(c2, "Average Churn Risk", pct(df["churn_probability"].mean()), "mean model probability",
        "warn" if df["churn_probability"].mean() > 0.3 else "good")
    kpi(c3, "High-Risk Accounts", f"{n_high:,}", f"{n_high / len(df):.1%} of active view", "bad")
    kpi(c4, "Monthly Revenue at Risk", money(df["expected_monthly_loss"].sum()), "probability x monthly charge")
    if "actual_churn" in df.columns and to_binary(df["actual_churn"]).notna().any():
        ar = to_binary(df["actual_churn"]).mean()
        kpi(c5, "Historical Churn Rate", pct(ar), "ground-truth labelled data", "warn")
    else:
        kpi(c5, "Avg Monthly Bill", money(df["MonthlyCharges"].mean()), "active cohort average")

    st.write("")
    a, b = st.columns(2)
    with a, st.container(border=True):
        card_title("Risk Band Distribution", "Customer breakdown across risk categories")
        fig = px.pie(df, names="risk_band", color="risk_band", color_discrete_map=BAND_COLORS,
                     category_orders={"risk_band": BAND_ORDER}, hole=0.62)
        fig.update_traces(textinfo="percent", sort=False)
        chart(style(fig, 320), key="ov_pie")
    with b, st.container(border=True):
        card_title("Revenue at Risk by Tier", "Expected annual revenue exposure")
        bd = by.reset_index()
        bd.columns = ["risk_band", "revenue"]
        fig = px.bar(bd, x="risk_band", y="revenue", color="risk_band", color_discrete_map=BAND_COLORS,
                     text=bd["revenue"].map(money), category_orders={"risk_band": BAND_ORDER})
        fig.update_layout(xaxis_title=None, yaxis_title="Revenue at Risk ($)")
        chart(style(fig, 320, legend=False), key="ov_bar")

    dims = [c for c in ["Contract", "InternetService", "PaymentMethod", "tenure_band", "TechSupport",
                        "OnlineSecurity", "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling", "gender"]
            if c in df.columns]
    if dims:
        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                card_title("Segment Explorer", "Analyze risk concentration and revenue exposure across customer segments")
            dim = top[1].selectbox("Group by", dims, label_visibility="collapsed")
            g = seg_table(df, dim)
            x1, x2 = st.columns(2)
            f1 = px.bar(g, x="avg_risk", y=dim, orientation="h", color="avg_risk",
                        color_continuous_scale=RISK_SCALE, text=g["avg_risk"].map(lambda v: f"{v:.0%}"),
                        hover_data={"customers": True})
            f1.update_layout(coloraxis_showscale=False, xaxis_title="Average Churn Risk", yaxis_title=None,
                             xaxis_tickformat=".0%")
            f1.update_xaxes(range=[0, max(g["avg_risk"].max() * 1.2, 0.05)])
            with x1:
                chart(style(f1, 300, legend=False), key="ov_seg1")
            f2 = px.bar(g.sort_values("revenue"), x="revenue", y=dim, orientation="h",
                        text=g.sort_values("revenue")["revenue"].map(money), color_discrete_sequence=[C["accent"]])
            f2.update_layout(xaxis_title="Annual Revenue at Risk ($)", yaxis_title=None)
            with x2:
                chart(style(f2, 300, legend=False), key="ov_seg2")

    if {"Contract", "tenure_band"} <= set(df.columns) or {"Contract", "InternetService"} <= set(df.columns):
        h, t = st.columns(2)
        if {"Contract", "tenure_band"} <= set(df.columns):
            with h, st.container(border=True):
                card_title("Risk Heatmap", "Average churn probability by contract structure and tenure cohort")
                pv = df.pivot_table(index="Contract", columns="tenure_band", values="churn_probability", aggfunc="mean")
                pv = pv.reindex(columns=[c for c in TENURE_LABELS if c in pv.columns])
                fig = px.imshow(pv, text_auto=".0%", aspect="auto", color_continuous_scale=RISK_SCALE)
                fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
                chart(style(fig, 320, legend=False), key="ov_heat")
        if {"Contract", "InternetService"} <= set(df.columns):
            with t, st.container(border=True):
                card_title("Revenue Concentration Treemap", "Exposure breakdown by contract, service type, and risk tier")
                tm = (df.assign(Contract=df["Contract"].astype(str), InternetService=df["InternetService"].astype(str),
                                risk_band=df["risk_band"].astype(str))
                      .groupby(["Contract", "InternetService", "risk_band"])["annual_revenue_at_risk"].sum().reset_index())
                tm = tm[tm["annual_revenue_at_risk"] > 0]
                fig = px.treemap(tm, path=[px.Constant("All Customers"), "Contract", "InternetService", "risk_band"],
                                 values="annual_revenue_at_risk", color="risk_band",
                                 color_discrete_map={**BAND_COLORS, "(?)": C["line"]})
                fig.update_traces(textinfo="label+value", texttemplate="%{label}<br>$%{value:,.0f}")
                chart(style(fig, 320, legend=False), key="ov_tree")

    if {"tenure", "MonthlyCharges"} <= set(df.columns):
        with st.container(border=True):
            card_title("Tenure vs. Monthly Charges", "Scatter distribution of accounts color-coded by model risk score")
            s = df.sample(min(len(df), 2500), random_state=1)
            hd = [c for c in ["customerID", "risk_band"] if c in s.columns]
            fig = px.scatter(s, x="tenure", y="MonthlyCharges", color="churn_probability",
                             color_continuous_scale=RISK_SCALE, opacity=0.75, hover_data=hd)
            fig.update_layout(coloraxis_colorbar=dict(title="Risk", tickformat=".0%"))
            chart(style(fig, 360, legend=False), key="ov_scatter")

    with st.container(border=True):
        card_title("Top 10 High-Exposure Accounts", "Priority outreach list ranked by annual revenue at risk")
        cols = [c for c in ["customerID", "risk_band", "churn_probability", "MonthlyCharges", "tenure",
                            "Contract", "annual_revenue_at_risk", "top_3_churn_reasons"] if c in df.columns]
        t10 = df.nlargest(10, "annual_revenue_at_risk")[cols].copy()
        t10["risk_band"] = t10["risk_band"].astype(str)
        table(t10, column_config=col_config(t10))


def col_config(d):
    cfg = {}
    if "churn_pct" in d.columns:
        cfg["churn_pct"] = st.column_config.ProgressColumn("Churn Risk", min_value=0, max_value=100, format="%.1f%%")
    if "churn_probability" in d.columns:
        cfg["churn_probability"] = st.column_config.NumberColumn("Probability", format="%.3f")
    for c in ["MonthlyCharges"]:
        if c in d.columns:
            cfg[c] = st.column_config.NumberColumn("Monthly Bill", format="$%.2f")
    if "annual_revenue_at_risk" in d.columns:
        cfg["annual_revenue_at_risk"] = st.column_config.NumberColumn("Annual Revenue at Risk", format="$%.0f")
    if "top_3_churn_reasons" in d.columns:
        cfg["top_3_churn_reasons"] = st.column_config.TextColumn("Key Drivers", width="large")
    return cfg


# ------------------------------------------------------------- EXPLORER
def page_explorer(df):
    header("Risk Explorer", "Filter, inspect individual customer profiles, and export cohort rosters to CSV.")
    c1, c2, c3 = st.columns([1.3, 1.3, 1.6])
    bands = c1.multiselect("Risk Band", BAND_ORDER, default=["High"])
    prange = c2.slider("Churn Probability Range", 0.0, 1.0, (0.0, 1.0), 0.01)
    q = c3.text_input("Search Customer ID", placeholder="e.g. 7590-VHVEG")

    v = df.copy()
    if bands:
        v = v[v["risk_band"].astype(str).isin(bands)]
    v = v[v["churn_probability"].between(*prange)]
    if q.strip():
        v = v[v["customerID"].astype(str).str.contains(q.strip(), case=False, na=False)]
    if v.empty:
        st.info("No customer records match the current criteria. Please broaden your probability range or risk tiers.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Customers", f"{len(v):,}")
    m2.metric("Average Churn Risk", pct(v["churn_probability"].mean()))
    m3.metric("Total Monthly Billings", money(v["MonthlyCharges"].sum()))
    m4.metric("Annual Revenue at Risk", money(v["annual_revenue_at_risk"].sum()))

    with st.container(border=True):
        card_title("Probability Distribution", "Dashed lines represent current risk band cutoffs")
        fig = px.histogram(v, x="churn_probability", color="risk_band", nbins=40,
                           color_discrete_map=BAND_COLORS, category_orders={"risk_band": BAND_ORDER})
        for x in (LO_CUT, HI_CUT):
            fig.add_vline(x=x, line_dash="dot", line_color=C["mute"])
        fig.update_layout(xaxis_title="Churn Probability", yaxis_title="Customer Count", xaxis_tickformat=".0%")
        chart(style(fig, 280), key="ex_hist")

    with st.container(border=True):
        card_title("Customer Roster", f"{len(v):,} accounts, sorted by descending risk probability")
        v = v.sort_values("churn_probability", ascending=False).copy()
        v["churn_pct"] = v["churn_probability"] * 100
        v["risk_band"] = v["risk_band"].astype(str)
        cols = [c for c in ["customerID", "risk_band", "churn_pct", "MonthlyCharges", "tenure", "Contract",
                            "annual_revenue_at_risk", "top_3_churn_reasons"] if c in v.columns]
        table(v[cols], column_config=col_config(v[cols]), height=460)
        st.download_button("Download CSV Roster", v.drop(columns=["churn_pct"]).to_csv(index=False).encode(),
                           "risk_roster.csv", "text/csv")


# ------------------------------------------------------------- CUSTOMER 360
def page_c360(df):
    header("Customer 360", "Comprehensive account profile: churn probability, financial exposure, and key risk drivers.")
    ids = df.sort_values("churn_probability", ascending=False)["customerID"].astype(str).tolist()
    cid = st.selectbox("Select Customer", ids, help="Sorted by churn risk score; type ID to filter")
    row = df[df["customerID"].astype(str) == cid].iloc[0]

    left, right = st.columns([1.25, 1])
    with left, st.container(border=True):
        card_title(f"Customer {cid}", f"Risk Tier: {row['risk_band']}")
        items = [(c, row[c]) for c in df.columns if c not in PROFILE_SKIP and pd.notna(row[c])]
        kv = "".join(f"<div><span>{k}</span><b>{v if not isinstance(v, float) else round(v, 2)}</b></div>"
                     for k, v in items[:18])
        st.markdown(clean(f'<div class="kv">{kv}</div>'), unsafe_allow_html=True)
        reasons = parse_reasons(row.get("top_3_churn_reasons"))
        if reasons:
            st.write("")
            card_title("Primary Risk Factors")
            st.markdown("".join(f'<span class="chip">{r}</span>' for r in reasons), unsafe_allow_html=True)
    with right, st.container(border=True):
        chart(gauge(float(row["churn_probability"]), title="Churn Risk", lo=LO_CUT, hi=HI_CUT), key="c360_g")
        k1, k2 = st.columns(2)
        k1.metric("Monthly Billing", money(row["MonthlyCharges"]))
        k2.metric("Annual Revenue at Risk", money(row["annual_revenue_at_risk"]))

    pct_rank = (df["churn_probability"] < row["churn_probability"]).mean()
    note(f"This account carries higher churn risk than <b>{pct_rank:.0%}</b> of the active customer base ({len(df):,} accounts).")


# ------------------------------------------------------------- WHAT-IF
def build_scenarios(base, opts):
    sc = {}
    cur = str(base.get("Contract", ""))
    best_contract = None
    for v in opts.get("Contract", []):
        if v != cur and "month" not in str(v).lower():
            sc[f"Move to {v} contract"] = {"Contract": v}
            best_contract = v
    for col, label in [("TechSupport", "Add Tech Support"), ("OnlineSecurity", "Add Online Security"),
                       ("OnlineBackup", "Add Online Backup"), ("DeviceProtection", "Add Device Protection")]:
        if col in base and "Yes" in opts.get(col, []) and str(base[col]) != "Yes":
            sc[label] = {col: "Yes"}
    for v in opts.get("PaymentMethod", []):
        if v != base.get("PaymentMethod") and "automatic" in str(v).lower():
            sc[f"Switch to {v}"] = {"PaymentMethod": v}
    if "MonthlyCharges" in base:
        sc["10% Price Discount"] = {"MonthlyCharges": round(float(base["MonthlyCharges"]) * 0.9, 2)}
    if best_contract and "TechSupport" in base and "Yes" in opts.get("TechSupport", []):
        sc[f"Bundle: {best_contract} + Tech Support"] = {"Contract": best_contract, "TechSupport": "Yes"}
    return sc


def page_whatif(df):
    header("What-If Simulator", "Simulate plan modifications and test real-time impact on predicted churn probability.")
    scope = st.radio("Cohort Scope", ["High risk only", "All customers in view"], horizontal=True)
    pool = df[df["risk_band"] == "High"] if scope == "High risk only" and (df["risk_band"] == "High").any() else df
    ids = pool.sort_values("churn_probability", ascending=False)["customerID"].astype(str).tolist()
    cid = st.selectbox("Select Customer", ids)
    base = pool[pool["customerID"].astype(str) == cid].iloc[0].to_dict()
    cur = float(base["churn_probability"])
    st.markdown(f"**Customer {cid}** &nbsp;|&nbsp; Current churn risk: **{cur:.1%}**")

    opts = {c: sorted(DF[c].dropna().astype(str).unique()) for c in
            ["Contract", "InternetService", "TechSupport", "OnlineSecurity", "OnlineBackup",
             "DeviceProtection", "PaymentMethod"] if c in DF.columns}
    left, right = st.columns(2)
    changes = {}
    with left, st.container(border=True):
        card_title("Configure Retention Intervention")
        for col, label in [("Contract", "Contract"), ("InternetService", "Internet Service"),
                           ("TechSupport", "Tech Support"), ("OnlineSecurity", "Online Security"),
                           ("PaymentMethod", "Payment Method")]:
            if col in opts:
                cur_v = str(base.get(col))
                idx = opts[col].index(cur_v) if cur_v in opts[col] else 0
                changes[col] = st.selectbox(label, opts[col], index=idx, key=f"wi_{col}_{cid}")
        if "MonthlyCharges" in base:
            lo, hi = float(DF["MonthlyCharges"].min()), float(DF["MonthlyCharges"].max())
            changes["MonthlyCharges"] = st.slider("Monthly Charges ($)", lo, hi,
                                                  float(np.clip(base["MonthlyCharges"], lo, hi)),
                                                  key=f"wi_mc_{cid}")
        run = st.button("Run Simulation", type="primary", **stretch(st.button))

    if run:
        try:
            new = predict_prob(base, changes)
            scen = {k: predict_prob(base, v) for k, v in build_scenarios(base, opts).items()}
            st.session_state["sim"] = dict(cid=cid, new=new, scen=scen, changes=changes)
        except Exception as e:
            st.error(f"Simulation execution failed: {e}")

    sim = st.session_state.get("sim")
    with right, st.container(border=True):
        card_title("Projected Outcome")
        if sim and sim["cid"] == cid:
            chart(gauge(sim["new"], ref=cur, title="Simulated Churn Risk", lo=LO_CUT, hi=HI_CUT), key="wi_g")
            mc = float(sim["changes"].get("MonthlyCharges", base["MonthlyCharges"]))
            old_rev, new_rev = cur * float(base["MonthlyCharges"]) * 12, sim["new"] * mc * 12
            r1, r2, r3 = st.columns(3)
            r1.metric("Risk Delta", f"{(sim['new'] - cur) * 100:+.1f} pts")
            r2.metric("Simulated Revenue at Risk", money(new_rev), money(new_rev - old_rev), delta_color="inverse")
            r3.metric("Projected Value Saved", money(max(old_rev - new_rev, 0)))
        else:
            st.info("Adjust the account parameters on the left and click 'Run Simulation' to evaluate projected outcomes.")

    if sim and sim["cid"] == cid and sim["scen"]:
        with st.container(border=True):
            card_title("Single-Action Sensitivity Analysis", "Impact of isolated interventions on customer retention probability")
            s = pd.DataFrame({"Action": list(sim["scen"].keys()),
                              "New risk": list(sim["scen"].values())})
            s["Risk change (pts)"] = (s["New risk"] - cur) * 100
            s = s.sort_values("Risk change (pts)")
            fig = px.bar(s, x="Risk change (pts)", y="Action", orientation="h",
                         color="Risk change (pts)", color_continuous_scale=[[0, C["low"]], [0.5, C["med"]], [1, C["high"]]],
                         text=s["Risk change (pts)"].map(lambda v: f"{v:+.1f}"))
            fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
            chart(style(fig, 80 + 40 * len(s), legend=False), key="wi_scen")
            note("Estimates reflect pipeline probability predictions. Validate real-world campaign elasticity via randomized A/B trials.")


# ------------------------------------------------------------- PLANNER
def page_planner(df):
    header("Retention Campaign Planner", "Model campaign unit economics and optimize outreach thresholds for maximum net profit.")
    c1, c2, c3 = st.columns(3)
    offer = c1.number_input("Offer Cost per Account ($)", 0.0, 1000.0, 50.0, 5.0)
    rate = c2.slider("Estimated Offer Acceptance Rate", 0.05, 1.0, 0.30, 0.05,
                     help="Expected proportion of churn-prone accounts saved by this campaign")
    months = c3.number_input("Customer Lifetime Saved (Months)", 1, 36, 12, 1)

    d = df[["customerID", "risk_band", "churn_probability", "MonthlyCharges"]
           + [c for c in ["top_3_churn_reasons"] if c in df.columns]].copy()
    d["value"] = d["churn_probability"] * rate * d["MonthlyCharges"] * months
    d["net"] = d["value"] - offer
    d = d.sort_values("net", ascending=False).reset_index(drop=True)
    d["cum_net"] = d["net"].cumsum()
    n_opt = int(d["cum_net"].idxmax() + 1) if d["cum_net"].max() > 0 else 0

    n = st.slider("Target Outreach Volume", 0, len(d), n_opt,
                  help=f"Optimal threshold: {n_opt:,} accounts (maximizes cumulative net profit)")
    sel = d.head(n)
    cost, gain = n * offer, float(sel["value"].sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Campaign Cost", money(cost))
    k2.metric("Gross Revenue Preserved", money(gain))
    k3.metric("Net Economic Benefit", money(gain - cost))
    k4.metric("Campaign ROI", f"{(gain - cost) / cost:.0%}" if cost > 0 else "n/a")

    with st.container(border=True):
        card_title("Net Return vs. Outreach Volume", "Outreach beyond the inflection point yields negative marginal returns")
        x = np.arange(1, len(d) + 1)
        fig = go.Figure(go.Scatter(x=x, y=d["cum_net"], mode="lines", line=dict(color=C["accent"], width=3)))
        if n > 0:
            fig.add_trace(go.Scatter(x=[n], y=[d["cum_net"].iloc[n - 1]], mode="markers",
                                     marker=dict(size=12, color=C["med"]), name="Selected Target"))
        fig.update_layout(xaxis_title="Outreach Volume (Ranked by Net Value)", yaxis_title="Cumulative Net Return ($)")
        chart(style(fig, 320, legend=False), key="pl_curve")

    with st.container(border=True):
        card_title("Priority Outreach Queue", "Rank-ordered intervention call list")
        out = sel.copy()
        out["risk_band"] = out["risk_band"].astype(str)
        out["churn_pct"] = out["churn_probability"] * 100
        cols = [c for c in ["customerID", "risk_band", "churn_pct", "MonthlyCharges", "value", "top_3_churn_reasons"]
                if c in out.columns]
        cfg = col_config(out)
        cfg["value"] = st.column_config.NumberColumn("Expected Saved ($)", format="$%.0f")
        table(out[cols].head(200), column_config=cfg, height=380)
        st.download_button("Export Campaign Roster (CSV)", out[cols].to_csv(index=False).encode(),
                           "campaign_outreach_list.csv", "text/csv")
    note("Optimization calculations rely on well-calibrated risk probabilities to avoid misallocating retention budgets.")


# ------------------------------------------------------------- MODEL HEALTH
def page_health(df):
    header("Model Diagnostics & Performance", "Evaluate discrimination, calibration accuracy, and optimal decision thresholds.")
    if "actual_churn" not in df.columns:
        st.info("This evaluation view requires an 'actual_churn' ground-truth column in risk_scores.csv.")
        return
    d = df.copy()
    if "data_split" in d.columns:
        splits = sorted(d["data_split"].dropna().astype(str).unique())
        default = [s for s in splits if s.lower() == "test"] or splits
        pick = st.multiselect("Dataset Partition", splits, default=default)
        d = d[d["data_split"].astype(str).isin(pick)]
        if any(p.lower() == "train" for p in pick):
            note("Training set metrics may exhibit optimistic bias. Evaluate against the Holdout/Test split for unbiased verification.")
    d["y"] = to_binary(d["actual_churn"])
    d = d.dropna(subset=["y", "churn_probability"])
    if d["y"].nunique() < 2:
        st.warning("Selected partition contains only one target class; performance curves cannot be computed.")
        return

    from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    y, p = d["y"].astype(int).values, d["churn_probability"].values
    a, b, c, e = st.columns(4)
    a.metric("ROC-AUC", f"{roc_auc_score(y, p):.3f}")
    b.metric("PR-AUC", f"{average_precision_score(y, p):.3f}", f"baseline: {y.mean():.3f}", delta_color="off")
    c.metric("Brier Score", f"{brier_score_loss(y, p):.3f}", "lower is better", delta_color="off")
    e.metric("Sample Size", f"{len(d):,}")

    t1, t2, t3, t4 = st.tabs(["Ranking Performance", "Calibration", "Cost-Sensitive Thresholds", "Decile Lift"])
    with t1:
        x1, x2 = st.columns(2)
        fpr, tpr, _ = roc_curve(y, p)
        f = go.Figure([go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=C["accent"], width=3), name="Model"),
                       go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C["mute"], dash="dot"), name="Random Guess")])
        f.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)")
        with x1:
            card_title("ROC Curve")
            chart(style(f, 320), key="h_roc")
        pr, rc, _ = precision_recall_curve(y, p)
        f = go.Figure([go.Scatter(x=rc, y=pr, mode="lines", line=dict(color=C["low"], width=3), name="Model"),
                       go.Scatter(x=[0, 1], y=[y.mean()] * 2, mode="lines", line=dict(color=C["mute"], dash="dot"), name="Baseline")])
        f.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        with x2:
            card_title("Precision-Recall Curve")
            chart(style(f, 320), key="h_pr")
    with t2:
        card_title("Reliability Diagram (Calibration Curve)", "Alignment with diagonal indicates true probabilistic accuracy")
        bins = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
        cal = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True).agg(
            predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()
        f = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C["mute"], dash="dot"), name="Ideal Calibration"),
                       go.Scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers",
                                  line=dict(color=C["accent"], width=3), marker=dict(size=cal["n"].clip(upper=400) / 25 + 6),
                                  name="Observed Calibration")])
        f.update_layout(xaxis_title="Mean Predicted Probability", yaxis_title="Empirical Churn Rate",
                        xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        chart(style(f, 360), key="h_cal")
    with t3:
        k1, k2, k3 = st.columns(3)
        fn_cost = k1.number_input("Cost of False Negative (Missed Churner) ($)", 0.0, 10000.0,
                                  float(round(d["MonthlyCharges"].mean() * 12)), 10.0)
        fp_cost = k2.number_input("Cost of False Positive (Wasted Incentive) ($)", 0.0, 10000.0, 150.0, 10.0)
        ths = np.round(np.arange(0.02, 0.99, 0.01), 2)
        costs = []
        for t in ths:
            pred = p >= t
            costs.append(((~pred) & (y == 1)).sum() * fn_cost + (pred & (y == 0)).sum() * fp_cost)
        best = float(ths[int(np.argmin(costs))])
        t_sel = k3.slider("Decision Threshold", 0.02, 0.98, best, 0.01)
        pred = p >= t_sel
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int(((~pred) & (y == 1)).sum()), int(((~pred) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        note(f"Optimal economic threshold: <b>{best:.2f}</b>. Selected threshold: <b>{t_sel:.2f}</b>.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recall", pct(rec)); m2.metric("Precision", pct(prec))
        m3.metric("F1 Score", f"{f1:.2f}"); m4.metric("Total Misclassification Cost", money(fn * fn_cost + fp * fp_cost))
        x1, x2 = st.columns(2)
        cm = px.imshow([[tn, fp], [fn, tp]], text_auto=True, aspect="auto",
                       x=["Predicted Retain", "Predicted Churn"], y=["Actual Retain", "Actual Churn"],
                       color_continuous_scale=[[0, "#0b1a29"], [1, C["accent"]]])
        cm.update_layout(coloraxis_showscale=False)
        with x1:
            card_title("Confusion Matrix")
            chart(style(cm, 300, legend=False), key="h_cm")
        f = go.Figure(go.Scatter(x=ths, y=costs, mode="lines", line=dict(color=C["med"], width=3)))
        f.add_vline(x=t_sel, line_dash="dot", line_color=C["accent"])
        f.update_layout(xaxis_title="Classification Threshold", yaxis_title="Total Business Cost ($)")
        with x2:
            card_title("Misclassification Cost Curve")
            chart(style(f, 300, legend=False), key="h_cost")
    with t4:
        card_title("Decile Lift Profile", "Churn concentration index relative to baseline incidence rate")
        d2 = pd.DataFrame({"p": p, "y": y})
        d2["decile"] = 10 - pd.qcut(d2["p"].rank(method="first"), 10, labels=False)
        lg = d2.groupby("decile").agg(churn_rate=("y", "mean"), n=("y", "size")).reset_index().sort_values("decile")
        lg["lift"] = lg["churn_rate"] / y.mean()
        f = px.bar(lg, x="decile", y="churn_rate", text=lg["lift"].map(lambda v: f"{v:.1f}x"),
                   color_discrete_sequence=[C["accent"]])
        f.add_hline(y=y.mean(), line_dash="dot", line_color=C["med"], annotation_text="Baseline Churn Rate")
        f.update_layout(xaxis_title="Risk Decile (1 = Highest Risk)", yaxis_title="Empirical Churn Rate", yaxis_tickformat=".0%",
                        xaxis=dict(dtick=1))
        chart(style(f, 340, legend=False), key="h_lift")


# ------------------------------------------------------------- SHAP
def page_shap(df):
    header("Model Interpretability & SHAP Analytics", "Global feature attributions and recurring churn drivers.")
    with st.container(border=True):
        card_title("Global SHAP Feature Summary", "Aggregate feature impact on model predictions")
        p = find_file(SHAP_FILES)
        if p:
            show_image(p)
        else:
            st.error("SHAP plot not found (reports/figures/shap_summary.png). Run your pipeline explanation notebook to generate and export the figure.")

    if "top_3_churn_reasons" in df.columns:
        rows = []
        for _, r in df.iterrows():
            for reason in parse_reasons(r["top_3_churn_reasons"]):
                rows.append((reason, r["annual_revenue_at_risk"], str(r["risk_band"])))
        if rows:
            rs = pd.DataFrame(rows, columns=["reason", "revenue", "band"])
            only_high = st.toggle("High-Risk Accounts Only", value=True)
            if only_high and (rs["band"] == "High").any():
                rs = rs[rs["band"] == "High"]
            g = rs.groupby("reason").agg(customers=("revenue", "size"), revenue=("revenue", "sum")).reset_index()
            g = g.sort_values("customers").tail(12)
            a, b = st.columns(2)
            with a, st.container(border=True):
                card_title("Most Prevalent Churn Drivers", "Account frequency where feature ranks in top 3 risk factors")
                f = px.bar(g, x="customers", y="reason", orientation="h", color_discrete_sequence=[C["high"]])
                f.update_layout(yaxis_title=None, xaxis_title="Account Count")
                chart(style(f, 380, legend=False), key="sh_c")
            with b, st.container(border=True):
                card_title("Revenue Exposure by Driver", "Total annual revenue linked to primary churn reasons")
                g2 = g.sort_values("revenue")
                f = px.bar(g2, x="revenue", y="reason", orientation="h", color_discrete_sequence=[C["accent"]],
                           text=g2["revenue"].map(money))
                f.update_layout(yaxis_title=None, xaxis_title="Revenue Exposure ($)")
                chart(style(f, 380, legend=False), key="sh_r")


# =====================================================================
# 6. ROUTER
# =====================================================================
ROUTES = {
    "overview": page_overview, "explorer": page_explorer, "c360": page_c360,
    "whatif": page_whatif, "planner": page_planner, "health": page_health, "shap": page_shap,
}
ROUTES[PAGES[page]](FDF)