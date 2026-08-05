from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_live_features, load_cleaned_data
from modeling import TARGETS, feature_importance_table, train_all_models

st.set_page_config(page_title="EduPro Predictive Analytics", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#0b1220; --panel:#121d31; --panel2:#16243b; --cyan:#4dd9ff; --blue:#5b8cff; --muted:#9aabc5; --line:rgba(157,181,220,.16); }
.stApp { background: radial-gradient(circle at top right, rgba(42,101,186,.24), transparent 35%), var(--bg); color:#f6f9ff; font-family:'DM Sans', sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:-.02em; }
.block-container { padding-top:3.2rem; padding-bottom:2rem; max-width:1500px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0f1b30 0%,#0a1220 100%); border-right:1px solid var(--line); }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(27,48,78,.96),rgba(17,31,52,.96)); border:1px solid var(--line); border-radius:18px; padding:18px 20px; box-shadow:0 12px 30px rgba(0,0,0,.18); }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:#fff; font-family:'Space Grotesk',sans-serif; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:16px; background:rgba(18,29,49,.72); }
div.stButton > button { border-radius:12px; border:1px solid rgba(77,217,255,.34); background:linear-gradient(135deg,#1d77c8,#2451b8); color:white; font-weight:600; }
.hero { padding:28px 32px; border-radius:24px; background:linear-gradient(120deg,rgba(29,73,128,.96),rgba(18,33,61,.96)); border:1px solid rgba(111,202,255,.25); box-shadow:0 20px 50px rgba(0,0,0,.22); margin-bottom:18px; }
.hero h1 { font-size:clamp(2rem,4vw,3.6rem); margin:0 0 8px; }
.hero p { color:#c5d7f5; max-width:850px; font-size:1.05rem; }
.badge { display:inline-block; margin:7px 6px 0 0; padding:7px 11px; border-radius:999px; background:rgba(77,217,255,.13); color:#bcefff; border:1px solid rgba(77,217,255,.25); font-size:.78rem; font-weight:600; }
.section { padding:20px 22px; border-radius:20px; background:rgba(18,29,49,.78); border:1px solid var(--line); box-shadow:0 10px 25px rgba(0,0,0,.12); margin:12px 0; }
.smallcaps { text-transform:uppercase; color:#70dfff; letter-spacing:.12em; font-size:.72rem; font-weight:700; }
.insight { padding:14px 16px; border-left:3px solid var(--cyan); background:rgba(77,217,255,.07); border-radius:10px; margin:8px 0; color:#dce9fb; }
</style>
"""

def init_page(section_label=None, title=None):
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()
    if section_label and title:
        st.markdown(f'<div class="section"><div class="smallcaps">{section_label}</div><h2>{title}</h2></div>', unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎓 EduPro AI")
        st.caption("Predictive analytics command center")
        st.divider()
        st.markdown("### About the project")
        st.write("EduPro turns course, instructor, learner, and transaction data into live demand and revenue forecasts for planning, pricing, and portfolio decisions.")
        st.divider()
        st.markdown("### Dataset information")
        st.metric("Transactions", f"{len(DATA):,}")
        st.metric("Courses", f"{DATA['CourseID'].nunique():,}")
        st.metric("Teachers", f"{DATA['TeacherID'].nunique():,}")
        st.divider()
        st.markdown("### Model information")
        st.write("Five regressors are benchmarked for each target. The lowest validation RMSE is automatically deployed.")
        st.divider()
        st.markdown("### Technology stack")
        st.write("Streamlit · scikit-learn · pandas · Plotly")
        st.divider()
        st.markdown("### Navigation")
        st.caption("Select a page above to move through the analytics pipeline.")

@st.cache_data
def load_data():
    return load_cleaned_data()

@st.cache_resource(show_spinner=False)
def train_cached(data_signature):
    return train_all_models(DATA)

def money(value):
    return f"${value:,.0f}"

def safe_options(series):
    return sorted([str(x) for x in series.dropna().unique()])

def chart_layout(fig, height=390):
    fig.update_layout(template='plotly_dark', height=height, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

DATA = load_data()
MODELS = train_cached(f"{len(DATA)}-{DATA['TransactionID'].nunique()}")
