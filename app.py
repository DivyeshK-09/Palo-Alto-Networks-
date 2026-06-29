import streamlit as st
import pandas as pd

from pipeline import build_pipeline, CLUSTER_ACTIONS
from charts import (
    CLUSTER_COLORS, GAP_COLORS, TRAIN_COLORS,
    # Tab 1
    fig_pca_scatter, fig_cluster_donut, fig_cluster_heatmap,
    fig_attrition_by_cluster, fig_cluster_by_dept,
    # Tab 2
    fig_gap_by_role, fig_gap_donut, fig_gap_box,
    fig_gap_by_dept, fig_salary_vs_gap,
    # Tab 3
    fig_retention_donut, fig_retention_by_role,
    fig_training_by_dept, fig_training_vs_attrition,
    # Tab 4
    fig_manager_tenure_vs_attrition, fig_manager_vs_gap,
    fig_stagnation_by_dept, fig_overtime_dual,
    fig_wlb_vs_gap, fig_jobsat_vs_attrition,
    # Tab 5
    fig_metric_histogram, fig_correlation_matrix,
)

st.set_page_config(
    page_title="Career Intelligence | Palo Alto Networks",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main, .stApp { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2d3555; border-radius: 12px;
        padding: 20px 24px; margin-bottom: 12px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #7c83ff; margin: 0; }
    .metric-label { font-size: 0.85rem; color: #8b92a9; margin: 0;
                    text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-sub   { font-size: 0.8rem; color: #5a6278; margin-top: 4px; }
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #e0e4ff;
        border-left: 4px solid #7c83ff; padding-left: 12px;
        margin: 24px 0 16px 0;
    }
    [data-testid="stSidebar"] { background: #13162b; border-right: 1px solid #1e2240; }
    .stTabs [data-baseweb="tab"] { background: #1e2130; border-radius: 8px; color: #8b92a9; padding: 8px 20px; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#7c83ff22,#5c63dd22);
                                      color: #7c83ff; border-bottom: 2px solid #7c83ff; }
    hr { border-color: #2d3555; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "Palo_Alto_Networks.csv"

@st.cache_data
def load_data(path):
    return build_pipeline(path)

try:
    df_full = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("⚠️  `Palo_Alto_Networks.csv` not found. Place it in the same directory as `app.py`.")
    st.stop()

with st.sidebar:
    st.markdown("## 🔭 Filters")
    st.markdown("---")

    sel_dept = st.selectbox("Department", ["All"] + sorted(df_full["Department"].unique().tolist()))
    sel_role = st.selectbox("Job Role",   ["All"] + sorted(df_full["JobRole"].unique().tolist()))

    career_stage = st.selectbox(
        "Career Stage",
        ["All", "Early (0-3 yrs)", "Mid (4-10 yrs)", "Senior (11+ yrs)"],
    )
    cluster_filter = st.multiselect(
        "Career Cluster", options=list(CLUSTER_COLORS.keys()),
        default=list(CLUSTER_COLORS.keys()),
    )
    promo_gap_threshold = st.slider("Max Years Since Last Promotion", 0, 15, 15)
    gap_filter = st.multiselect("Promotion Gap Risk", ["High","Medium","Low"],
                                default=["High","Medium","Low"])

    st.markdown("---")
    st.caption("Career Intelligence Dashboard v1.0\nPalo Alto Networks × Unified Mentor")

df = df_full.copy()
if sel_dept != "All":      df = df[df["Department"] == sel_dept]
if sel_role != "All":      df = df[df["JobRole"] == sel_role]
if career_stage == "Early (0-3 yrs)":    df = df[df["YearsAtCompany"] <= 3]
elif career_stage == "Mid (4-10 yrs)":   df = df[(df["YearsAtCompany"] >= 4) & (df["YearsAtCompany"] <= 10)]
elif career_stage == "Senior (11+ yrs)": df = df[df["YearsAtCompany"] >= 11]
if cluster_filter: df = df[df["ClusterLabel"].isin(cluster_filter)]
df = df[df["YearsSinceLastPromotion"] <= promo_gap_threshold]
df = df[df["PromotionGapScore"].isin(gap_filter)]

st.markdown("""
<div style="background:linear-gradient(135deg,#1a1e35,#1e2440);
     border:1px solid #2d3555;border-radius:16px;padding:28px 32px;margin-bottom:24px;">
  <div style="display:flex;align-items:center;gap:16px;">
    <span style="font-size:2.4rem;">🚀</span>
    <div>
      <h1 style="margin:0;font-size:1.8rem;color:#e0e4ff;font-weight:700;">
        Career Progression &amp; Promotion Gap Intelligence
      </h1>
      <p style="margin:4px 0 0 0;color:#8b92a9;font-size:0.92rem;">
        Palo Alto Networks — Retention Optimization Platform | Unified Mentor
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

def kpi(col, value, label, sub=""):
    col.markdown(f"""
    <div class="metric-card">
      <p class="metric-label">{label}</p>
      <p class="metric-value">{value}</p>
      <p class="metric-sub">{sub}</p>
    </div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpi(c1, f"{len(df):,}",                                    "Employees",          "filtered view")
kpi(c2, f"{(df['PromotionGapScore']=='High').sum():,}",     "High Gap Risk",      "promotion stagnated")
kpi(c3, f"{(df['RetentionOpportunityIndex']=='High Priority').sum():,}", "Retention Priority", "intervention needed")
kpi(c4, f"{df['Attrition'].mean()*100:.1f}%",              "Attrition Rate",     "left organisation")
kpi(c5, f"{df['YearsSinceLastPromotion'].mean():.1f} yrs",  "Avg Promo Gap",      "since last promotion")
kpi(c6, f"{(df['TrainingNeedIndicator']=='Critical').sum():,}", "Critical Training", "zero training last yr")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗂 Career Clustering",
    "📈 Promotion Gap Monitor",
    "🎯 Retention Opportunities",
    "🧑‍💼 Managerial Insights",
    "📋 Employee Explorer",
])

with tab1:
    st.markdown('<div class="section-header">Career Path Clustering</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])
    with col_l:  st.plotly_chart(fig_pca_scatter(df),       width="stretch")
    with col_r:  st.plotly_chart(fig_cluster_donut(df),     width="stretch")

    st.markdown('<div class="section-header">Cluster Profile Summary</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_cluster_heatmap(df), width="stretch")

    col_a, col_b = st.columns(2)
    with col_a: st.plotly_chart(fig_attrition_by_cluster(df), width="stretch")
    with col_b: st.plotly_chart(fig_cluster_by_dept(df),      width="stretch")

with tab2:
    st.markdown('<div class="section-header">Promotion Gap Monitor</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])
    with col_l: st.plotly_chart(fig_gap_by_role(df),  width="stretch")
    with col_r: st.plotly_chart(fig_gap_donut(df),    width="stretch")

    col_a, col_b = st.columns(2)
    with col_a: st.plotly_chart(fig_gap_box(df),      width="stretch")
    with col_b: st.plotly_chart(fig_gap_by_dept(df),  width="stretch")

    st.markdown('<div class="section-header">Salary Hike vs Promotion Gap</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_salary_vs_gap(df), width="stretch")

    st.markdown('<div class="section-header">High Promotion Gap Employees</div>', unsafe_allow_html=True)
    high_gap_df = df[df["PromotionGapScore"] == "High"][[
        "Department","JobRole","JobLevel","YearsAtCompany",
        "YearsSinceLastPromotion","PercentSalaryHike",
        "PromotionGapRatio","RoleStagnationIndex","ClusterLabel",
    ]].sort_values("YearsSinceLastPromotion", ascending=False)
    st.dataframe(
        high_gap_df.reset_index(drop=True),
        width="stretch", height=320,
    )

with tab3:
    st.markdown('<div class="section-header">Retention Opportunity Panel</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 2])
    with col_l: st.plotly_chart(fig_retention_donut(df),   width="stretch")
    with col_r: st.plotly_chart(fig_retention_by_role(df), width="stretch")

    st.markdown('<div class="section-header">Training Need Indicator</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a: st.plotly_chart(fig_training_by_dept(df),       width="stretch")
    with col_b: st.plotly_chart(fig_training_vs_attrition(df),  width="stretch")

    st.markdown('<div class="section-header">🎯 Suggested Retention Actions</div>', unsafe_allow_html=True)
    for cluster, (badge, action) in CLUSTER_ACTIONS.items():
        count = len(df[df["ClusterLabel"] == cluster])
        if count > 0:
            col1, col2, col3 = st.columns([2,1,5])
            col1.markdown(f"**{cluster}**")
            col2.markdown(f"*{count} employees*")
            col3.markdown(action)

with tab4:
    st.markdown('<div class="section-header">Manager Stability & Team Career Health</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l: st.plotly_chart(fig_manager_tenure_vs_attrition(df), width="stretch")
    with col_r: st.plotly_chart(fig_manager_vs_gap(df),              width="stretch")

    st.markdown('<div class="section-header">Team-Level Stagnation Signals</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a: st.plotly_chart(fig_stagnation_by_dept(df), width="stretch")
    with col_b: st.plotly_chart(fig_overtime_dual(df),      width="stretch")

    col_c, col_d = st.columns(2)
    with col_c: st.plotly_chart(fig_wlb_vs_gap(df),            width="stretch")
    with col_d: st.plotly_chart(fig_jobsat_vs_attrition(df),   width="stretch")

with tab5:
    st.markdown('<div class="section-header">Employee-Level Career Explorer</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Years in Current Role", f"{df['YearsInCurrentRole'].mean():.1f}")
    col2.metric("Avg Salary Hike %",         f"{df['PercentSalaryHike'].mean():.1f}%")
    col3.metric("Avg Career Velocity",        f"{df['CareerVelocity'].mean():.3f}")

    st.markdown("##### Detailed Employee Career Records")
    display_cols = [
        "Department","JobRole","Age","JobLevel","YearsAtCompany",
        "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager",
        "TrainingTimesLastYear","PercentSalaryHike","MonthlyIncome",
        "PromotionGapScore","RetentionOpportunityIndex","TrainingNeedIndicator",
        "ClusterLabel","Attrition",
    ]
    st.dataframe(df[display_cols].reset_index(drop=True), width="stretch", height=480)

    st.markdown('<div class="section-header">Career Metric Distributions</div>', unsafe_allow_html=True)
    metric_select = st.selectbox("Select metric to explore", [
        "YearsSinceLastPromotion","YearsInCurrentRole","YearsAtCompany",
        "TrainingTimesLastYear","PercentSalaryHike","PromotionGapRatio",
        "RoleStagnationIndex","CareerVelocity",
    ])
    st.plotly_chart(fig_metric_histogram(df, metric_select), width="stretch")

    st.markdown('<div class="section-header">Career Feature Correlation Matrix</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_correlation_matrix(df), width="stretch")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#3d4466;font-size:0.78rem;'>"
    "Career Progression & Promotion Gap Intelligence Dashboard · Palo Alto Networks × Unified Mentor · 2025"
    "</p>", unsafe_allow_html=True,
)
