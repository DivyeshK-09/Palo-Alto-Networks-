# Career Progression & Promotion Gap Analysis Dashboard
**Palo Alto Networks × Unified Mentor**

## Setup & Run

### 1. Create and activate the virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

Or just use the convenience script (macOS/Linux):
```bash
./activate_and_run.sh
```

---

## Project Structure

```
rakshak_career_dashboard/
├── app.py               # Streamlit UI — layout, filters, tabs
├── pipeline.py          # Feature engineering, clustering, KPI scoring
├── charts.py            # All Plotly figure builders
├── requirements.txt     # Dependencies
├── activate_and_run.sh  # One-shot run script (macOS/Linux)
├── venv/                # Virtual environment (not committed to git)
└── Palo_Alto_Networks.csv
```

---

## Dashboard Modules

| Tab | Description |
|-----|-------------|
| 🗂 Career Clustering | PCA 2D scatter, cluster distribution, heatmap, attrition by cluster |
| 📈 Promotion Gap Monitor | Gap risk by role/dept, salary vs gap scatter, high-gap employee table |
| 🎯 Retention Opportunities | Priority panel, training need, suggested actions per cluster |
| 🧑‍💼 Managerial Insights | Manager tenure vs attrition, stagnation signals, overtime analysis |
| 📋 Employee Explorer | Filterable table, metric distributions, correlation matrix |

## Engineered Features

| Feature | Formula |
|---------|---------|
| PromotionGapRatio | YearsSinceLastPromotion / (YearsAtCompany + 1) |
| RoleStagnationIndex | YearsInCurrentRole / (YearsAtCompany + 1) |
| TrainingIntensityScore | TrainingTimesLastYear / (YearsAtCompany + 1) |
| ManagerStabilityIndicator | YearsWithCurrManager / (YearsAtCompany + 1) |
| CareerVelocity | JobLevel / (TotalWorkingYears + 1) |
| SalaryGrowthPerYear | PercentSalaryHike / (YearsAtCompany + 1) |

## Cluster Labels

| Cluster | Profile |
|---------|---------|
| Promotion-Stalled | ~5 yrs since promotion, high gap ratio, mid-tenure |
| Early-Career Explorers | Low tenure, recent promotion, active movers |
| Stable Long-Term | 10+ yrs, high role stagnation but low attrition |
| Senior Veterans | 20+ yrs, high job level, longest since promotion |
| New Joiners at Risk | <1 yr tenure, 35% attrition — highest risk group |
