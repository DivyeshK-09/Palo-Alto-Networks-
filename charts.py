"""
charts.py
Career Progression & Promotion Gap Analysis — Palo Alto Networks
All Plotly figure builders. Each function takes a DataFrame and returns a go.Figure.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Color Palettes ───────────────────────────────────────────────────────────

CLUSTER_COLORS = {
    "Promotion-Stalled":       "#ff4d6d",
    "Early-Career Explorers":  "#7c83ff",
    "Stable Long-Term":        "#22cc88",
    "Senior Veterans":         "#ffa022",
    "New Joiners at Risk":     "#ff6b9d",
}
GAP_COLORS    = {"High": "#ff4d6d", "Medium": "#ffa022", "Low": "#22cc88"}
RETAIN_COLORS = {"High Priority": "#ff4d6d", "Medium Priority": "#ffa022", "Low Priority": "#22cc88"}
TRAIN_COLORS  = {"Critical": "#ff4d6d", "High": "#ffa022", "Medium": "#7c83ff", "Low": "#22cc88"}

BASE = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font_color="#c8cde0",
    font_family="Inter, sans-serif",
    title_font_color="#e0e4ff",
    legend_bgcolor="#1e2130",
    legend_bordercolor="#2d3555",
)

GRID = dict(gridcolor="#1e2240")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Career Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def fig_pca_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="PCA1", y="PCA2",
        color="ClusterLabel", color_discrete_map=CLUSTER_COLORS,
        hover_data={"JobRole": True, "Department": True,
                    "YearsAtCompany": True, "YearsSinceLastPromotion": True,
                    "PCA1": False, "PCA2": False},
        title="Career Trajectory Clusters (PCA 2D Projection)",
        labels={"ClusterLabel": "Career Archetype"},
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(**BASE, height=460,
                      xaxis=dict(showgrid=False, zeroline=False, title="Principal Component 1"),
                      yaxis=dict(showgrid=False, zeroline=False, title="Principal Component 2"))
    return fig


def fig_cluster_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["ClusterLabel"].value_counts().reset_index()
    counts.columns = ["ClusterLabel", "Count"]
    fig = px.pie(counts, values="Count", names="ClusterLabel",
                 color="ClusterLabel", color_discrete_map=CLUSTER_COLORS,
                 hole=0.55, title="Cluster Distribution")
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(**BASE, height=460, showlegend=True,
                      legend=dict(orientation="v", x=1.0, y=0.5))
    return fig


def fig_cluster_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = ["YearsAtCompany", "YearsSinceLastPromotion", "YearsInCurrentRole",
            "JobLevel", "TrainingTimesLastYear", "PromotionGapRatio",
            "RoleStagnationIndex", "CareerVelocity"]
    profile = df.groupby("ClusterLabel")[cols].mean().round(2)
    norm    = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig = go.Figure(data=go.Heatmap(
        z=norm.values,
        x=[c.replace("Years", "Yrs ") for c in norm.columns],
        y=norm.index.tolist(),
        colorscale="Viridis",
        text=profile.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 11},
        showscale=True,
    ))
    fig.update_layout(**BASE, height=300,
                      title="Cluster Attribute Profiles (raw values shown, color = normalised intensity)",
                      margin=dict(l=180, r=20, t=50, b=60))
    return fig


def fig_attrition_by_cluster(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("ClusterLabel")["Attrition"].mean().mul(100).reset_index()
    data.columns = ["ClusterLabel", "AttritionRate"]
    data = data.sort_values("AttritionRate", ascending=False)
    fig = px.bar(data, x="ClusterLabel", y="AttritionRate",
                 color="ClusterLabel", color_discrete_map=CLUSTER_COLORS,
                 title="Attrition Rate by Career Cluster (%)",
                 labels={"AttritionRate": "Attrition %", "ClusterLabel": ""},
                 text="AttritionRate")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(**BASE, height=340, showlegend=False, yaxis=GRID)
    return fig


def fig_cluster_by_dept(df: pd.DataFrame) -> go.Figure:
    data = df.groupby(["Department", "ClusterLabel"]).size().reset_index(name="Count")
    fig = px.bar(data, x="Department", y="Count",
                 color="ClusterLabel", color_discrete_map=CLUSTER_COLORS,
                 barmode="stack", title="Career Clusters by Department",
                 labels={"Count": "Employees", "Department": ""})
    fig.update_layout(**BASE, height=340, yaxis=GRID)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Promotion Gap Monitor
# ═══════════════════════════════════════════════════════════════════════════════

def fig_gap_by_role(df: pd.DataFrame) -> go.Figure:
    data = df.groupby(["JobRole", "PromotionGapScore"]).size().reset_index(name="Count")
    fig = px.bar(data, x="JobRole", y="Count",
                 color="PromotionGapScore", color_discrete_map=GAP_COLORS,
                 barmode="stack", title="Promotion Gap Risk by Job Role",
                 labels={"Count": "Employees", "JobRole": ""},
                 category_orders={"PromotionGapScore": ["Low", "Medium", "High"]})
    fig.update_layout(**BASE, height=380, xaxis_tickangle=-30, yaxis=GRID)
    return fig


def fig_gap_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["PromotionGapScore"].value_counts().reset_index()
    counts.columns = ["Score", "Count"]
    fig = px.pie(counts, values="Count", names="Score",
                 color="Score", color_discrete_map=GAP_COLORS,
                 hole=0.6, title="Promotion Gap Score Distribution")
    fig.update_layout(**BASE, height=380)
    return fig


def fig_gap_box(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x="JobLevel", y="YearsSinceLastPromotion",
                 color="PromotionGapScore", color_discrete_map=GAP_COLORS,
                 title="Promotion Gap Distribution by Job Level",
                 labels={"YearsSinceLastPromotion": "Yrs Since Promotion", "JobLevel": "Job Level"},
                 category_orders={"PromotionGapScore": ["Low", "Medium", "High"]})
    fig.update_layout(**BASE, height=360, yaxis=GRID)
    return fig


def fig_gap_by_dept(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("Department")["YearsSinceLastPromotion"].mean().reset_index()
    data.columns = ["Department", "AvgGap"]
    data = data.sort_values("AvgGap", ascending=False)
    fig = px.bar(data, x="Department", y="AvgGap",
                 color="Department",
                 color_discrete_sequence=["#7c83ff", "#ff4d6d", "#22cc88"],
                 title="Average Years Since Last Promotion by Dept",
                 text="AvgGap",
                 labels={"AvgGap": "Avg Yrs Since Promotion", "Department": ""})
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(**BASE, height=360, showlegend=False, yaxis=GRID)
    return fig


def fig_salary_vs_gap(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="YearsSinceLastPromotion", y="PercentSalaryHike",
                     color="PromotionGapScore", color_discrete_map=GAP_COLORS,
                     size="YearsAtCompany", opacity=0.65,
                     hover_data=["JobRole", "Department", "JobLevel"],
                     title="Salary Hike vs Years Since Last Promotion (size = tenure)",
                     labels={"YearsSinceLastPromotion": "Years Since Last Promotion",
                             "PercentSalaryHike": "Salary Hike %"})
    fig.update_layout(**BASE, height=380, xaxis=GRID, yaxis=GRID)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Retention Opportunities
# ═══════════════════════════════════════════════════════════════════════════════

def fig_retention_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["RetentionOpportunityIndex"].value_counts().reset_index()
    counts.columns = ["Priority", "Count"]
    fig = px.pie(counts, values="Count", names="Priority",
                 color="Priority", color_discrete_map=RETAIN_COLORS,
                 hole=0.55, title="Retention Opportunity Index")
    fig.update_layout(**BASE, height=360)
    return fig


def fig_retention_by_role(df: pd.DataFrame) -> go.Figure:
    data = df.groupby(["JobRole", "RetentionOpportunityIndex"]).size().reset_index(name="Count")
    fig = px.bar(data, x="Count", y="JobRole",
                 color="RetentionOpportunityIndex", color_discrete_map=RETAIN_COLORS,
                 barmode="stack", orientation="h",
                 title="Retention Priority by Job Role",
                 labels={"Count": "Employees", "JobRole": ""},
                 category_orders={"RetentionOpportunityIndex": ["High Priority","Medium Priority","Low Priority"]})
    fig.update_layout(**BASE, height=360, xaxis=GRID)
    return fig


def fig_training_by_dept(df: pd.DataFrame) -> go.Figure:
    data = df.groupby(["Department", "TrainingNeedIndicator"]).size().reset_index(name="Count")
    fig = px.bar(data, x="Department", y="Count",
                 color="TrainingNeedIndicator", color_discrete_map=TRAIN_COLORS,
                 barmode="group", title="Training Need by Department",
                 category_orders={"TrainingNeedIndicator": ["Critical","High","Medium","Low"]})
    fig.update_layout(**BASE, height=340, yaxis=GRID)
    return fig


def fig_training_vs_attrition(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("TrainingNeedIndicator").agg(
        AttritionRate=("Attrition", "mean"),
    ).mul(100).reset_index()
    fig = px.bar(data.sort_values("AttritionRate", ascending=False),
                 x="TrainingNeedIndicator", y="AttritionRate",
                 color="TrainingNeedIndicator", color_discrete_map=TRAIN_COLORS,
                 text="AttritionRate",
                 title="Attrition Rate by Training Need Level",
                 labels={"AttritionRate": "Attrition %", "TrainingNeedIndicator": ""})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(**BASE, height=340, showlegend=False, yaxis=GRID)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Managerial Insights
# ═══════════════════════════════════════════════════════════════════════════════

def fig_manager_tenure_vs_attrition(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["ManagerTenureBin"] = pd.cut(
        df["YearsWithCurrManager"],
        bins=[-1, 1, 3, 6, 10, 40],
        labels=["<1 yr", "1-3 yrs", "3-6 yrs", "6-10 yrs", "10+ yrs"],
    )
    data = df.groupby("ManagerTenureBin", observed=True)["Attrition"].mean().mul(100).reset_index()
    data.columns = ["ManagerTenure", "AttritionRate"]
    fig = px.line(data, x="ManagerTenure", y="AttritionRate", markers=True,
                  title="Attrition Rate vs Manager Tenure",
                  labels={"AttritionRate": "Attrition %", "ManagerTenure": "Years with Current Manager"})
    fig.update_traces(line_color="#7c83ff", marker_color="#ff4d6d", marker_size=10)
    fig.update_layout(**BASE, height=340, yaxis=GRID, xaxis=GRID)
    return fig


def fig_manager_vs_gap(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="YearsWithCurrManager", y="YearsSinceLastPromotion",
                     color="ClusterLabel", color_discrete_map=CLUSTER_COLORS,
                     opacity=0.6,
                     title="Manager Tenure vs Promotion Gap",
                     labels={"YearsWithCurrManager": "Years with Current Manager",
                             "YearsSinceLastPromotion": "Years Since Last Promotion"},
                     hover_data=["Department", "JobRole"])
    fig.update_layout(**BASE, height=340, xaxis=GRID, yaxis=GRID)
    return fig


def fig_stagnation_by_dept(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby("Department").agg(
        AvgRoleStagnation=("RoleStagnationIndex", "mean"),
        AvgPromotionGap=("PromotionGapRatio", "mean"),
        AvgCareerVelocity=("CareerVelocity", "mean"),
        AvgManagerStability=("ManagerStabilityIndicator", "mean"),
    ).round(3).reset_index()
    melted = agg.melt(id_vars="Department", var_name="Metric", value_name="Score")
    fig = px.bar(melted, x="Metric", y="Score", color="Department", barmode="group",
                 title="Stagnation Signals by Department",
                 labels={"Score": "Index Value", "Metric": ""},
                 color_discrete_sequence=["#7c83ff", "#ff4d6d", "#22cc88"])
    fig.update_layout(**BASE, height=360, xaxis_tickangle=-20, yaxis=GRID)
    return fig


def fig_overtime_dual(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("OverTime").agg(
        AvgYearsSincePromotion=("YearsSinceLastPromotion", "mean"),
        AttritionRate=("Attrition", "mean"),
    ).round(2).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=data["OverTime"], y=data["AvgYearsSincePromotion"],
               name="Avg Yrs Since Promotion", marker_color="#7c83ff"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=data["OverTime"], y=data["AttritionRate"] * 100,
                   name="Attrition %", mode="markers+lines",
                   marker=dict(color="#ff4d6d", size=12), line=dict(color="#ff4d6d")),
        secondary_y=True,
    )
    fig.update_layout(**BASE, height=360,
                      title="Overtime: Promotion Gap & Attrition",
                      yaxis=dict(title="Avg Years Since Promotion", **GRID),
                      yaxis2=dict(title="Attrition %"),
                      legend=dict(x=0.6, y=0.95))
    return fig


def fig_wlb_vs_gap(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("WorkLifeBalance")["YearsSinceLastPromotion"].mean().reset_index()
    data["WorkLifeBalance"] = data["WorkLifeBalance"].map({1:"Poor",2:"Fair",3:"Good",4:"Excellent"})
    fig = px.bar(data, x="WorkLifeBalance", y="YearsSinceLastPromotion",
                 color="WorkLifeBalance",
                 color_discrete_sequence=["#ff4d6d","#ffa022","#7c83ff","#22cc88"],
                 title="Work-Life Balance vs Avg Promotion Gap",
                 labels={"YearsSinceLastPromotion": "Avg Yrs Since Promotion", "WorkLifeBalance": ""},
                 text="YearsSinceLastPromotion")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(**BASE, height=320, showlegend=False, yaxis=GRID)
    return fig


def fig_jobsat_vs_attrition(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("JobSatisfaction")["Attrition"].mean().mul(100).reset_index()
    data["JobSatisfaction"] = data["JobSatisfaction"].map({1:"Low",2:"Medium",3:"High",4:"Very High"})
    fig = px.bar(data, x="JobSatisfaction", y="Attrition",
                 color="JobSatisfaction",
                 color_discrete_sequence=["#ff4d6d","#ffa022","#7c83ff","#22cc88"],
                 title="Job Satisfaction vs Attrition Rate",
                 labels={"Attrition": "Attrition %", "JobSatisfaction": ""},
                 text="Attrition")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(**BASE, height=320, showlegend=False, yaxis=GRID)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Employee Explorer
# ═══════════════════════════════════════════════════════════════════════════════

def fig_metric_histogram(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.histogram(df, x=metric, color="ClusterLabel",
                       color_discrete_map=CLUSTER_COLORS,
                       barmode="overlay", opacity=0.7, nbins=30,
                       title=f"Distribution of {metric} by Career Cluster",
                       labels={metric: metric, "count": "Employees"})
    fig.update_layout(**BASE, height=380, yaxis=GRID, xaxis=GRID)
    return fig


def fig_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    cols = ["YearsAtCompany", "YearsSinceLastPromotion", "YearsInCurrentRole",
            "YearsWithCurrManager", "JobLevel", "TrainingTimesLastYear",
            "PercentSalaryHike", "MonthlyIncome", "PromotionGapRatio",
            "RoleStagnationIndex", "CareerVelocity", "Attrition"]
    corr = df[cols].corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont={"size": 9}, showscale=True,
    ))
    fig.update_layout(**BASE, height=520,
                      title="Correlation Matrix — Career & Attrition Features",
                      margin=dict(l=160, r=20, t=50, b=120),
                      xaxis=dict(tickangle=-40))
    return fig
