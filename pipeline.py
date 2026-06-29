import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

CLUSTER_FEATURES = [
    "PromotionGapRatio", "RoleStagnationIndex", "TrainingIntensityScore",
    "ManagerStabilityIndicator", "CareerVelocity", "YearsAtCompany",
    "YearsSinceLastPromotion", "YearsInCurrentRole", "JobLevel", "TrainingTimesLastYear",
]

CLUSTER_LABELS = {
    0: "Promotion-Stalled",
    1: "Early-Career Explorers",
    2: "Stable Long-Term",
    3: "Senior Veterans",
    4: "New Joiners at Risk",
}

CLUSTER_ACTIONS = {
    "Promotion-Stalled":      ("🔴 Promotion Review",    "Immediate promotion eligibility review. Schedule 1:1 with manager and HR."),
    "New Joiners at Risk":    ("🔴 Onboarding Support",  "Strengthen 90-day onboarding. Assign mentors. Check role-fit alignment."),
    "Early-Career Explorers": ("🟡 Growth Path Design",  "Define clear promotion timeline. Offer stretch projects and skill tracks."),
    "Stable Long-Term":       ("🟢 Recognition Program", "Recognise loyalty with non-monetary rewards. Explore lateral enrichment."),
    "Senior Veterans":        ("🟡 Succession Planning", "Engage in mentorship roles. Explore leadership or advisory tracks."),
}

N_CLUSTERS = 5
RANDOM_STATE = 42

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PromotionGapRatio"]         = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["RoleStagnationIndex"]       = df["YearsInCurrentRole"]      / (df["YearsAtCompany"] + 1)
    df["TrainingIntensityScore"]    = df["TrainingTimesLastYear"]   / (df["YearsAtCompany"] + 1)
    df["ManagerStabilityIndicator"] = df["YearsWithCurrManager"]    / (df["YearsAtCompany"] + 1)
    df["CareerVelocity"]            = df["JobLevel"]                / (df["TotalWorkingYears"] + 1)
    df["SalaryGrowthPerYear"]       = df["PercentSalaryHike"]       / (df["YearsAtCompany"] + 1)
    return df

def run_clustering(df: pd.DataFrame):
    """Fit scaler + KMeans, return (df_with_clusters, scaler, model, pca)."""
    X_raw = df[CLUSTER_FEATURES].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["Cluster"]      = km.fit_predict(X)
    df["ClusterLabel"] = df["Cluster"].map(CLUSTER_LABELS)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    df["PCA1"] = coords[:, 0]
    df["PCA2"] = coords[:, 1]

    return df, scaler, km, pca


def score_promotion_gap(row) -> str:
    score = 0
    if row["YearsSinceLastPromotion"] >= 5:   score += 2
    elif row["YearsSinceLastPromotion"] >= 3:  score += 1
    if row["PromotionGapRatio"] >= 0.5:        score += 2
    elif row["PromotionGapRatio"] >= 0.25:     score += 1
    if row["RoleStagnationIndex"] >= 0.7:      score += 1
    if row["PercentSalaryHike"] <= 12:         score += 1
    return "High" if score >= 4 else ("Medium" if score >= 2 else "Low")


def score_retention_opportunity(row) -> str:
    stagnation = (row["YearsSinceLastPromotion"] >= 3 or row["RoleStagnationIndex"] >= 0.5)
    stayed     = row["Attrition"] == 0
    tenure     = row["YearsAtCompany"] >= 2
    if stagnation and stayed and tenure:       return "High Priority"
    elif stayed and row["YearsSinceLastPromotion"] >= 2: return "Medium Priority"
    return "Low Priority"


def score_training_need(row) -> str:
    if row["TrainingTimesLastYear"] == 0:                              return "Critical"
    if row["TrainingTimesLastYear"] <= 1 and row["YearsInCurrentRole"] >= 3: return "High"
    if row["TrainingIntensityScore"] < 0.2:                            return "Medium"
    return "Low"


def apply_kpi_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PromotionGapScore"]        = df.apply(score_promotion_gap,          axis=1)
    df["RetentionOpportunityIndex"]= df.apply(score_retention_opportunity,  axis=1)
    df["TrainingNeedIndicator"]    = df.apply(score_training_need,          axis=1)
    return df


def build_pipeline(path: str) -> pd.DataFrame:
    """Load CSV → engineer features → cluster → KPI scores. Returns enriched DataFrame."""
    df = pd.read_csv(path)
    df = engineer_features(df)
    df, _, _, _ = run_clustering(df)
    df = apply_kpi_scores(df)
    return df
