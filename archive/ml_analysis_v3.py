import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from src.database import AuctionDatabase

# Connect to database
db = AuctionDatabase()

# Pull weekly data with geographic breakdown
print("Fetching data with geographic dimensions...")

# Get base weekly data
base_data = db.query("""
    SELECT
        fiscal_year,
        fiscal_week_number,
        avg_lot_value,
        goal_status,
        construction_pct,
        construction_alv,
        ag_pct,
        ag_alv,
        passenger_pct,
        passenger_alv,
        trucks_med_heavy_pct,
        trucks_med_heavy_alv,
        total_items_sold
    FROM weekly_metrics_summary_enrichedv2
    WHERE has_auctions = true
""")

# Get regional breakdown per week
region_data = db.query("""
    SELECT
        fiscal_year,
        fiscal_week_number,
        item_region_id,
        total_items_sold,
        avg_lot_value,
        total_contract_price
    FROM weekly_metrics_by_regionv2
""")

db.close()

# Convert to DataFrames
df_base = pd.DataFrame(base_data)
df_region = pd.DataFrame(region_data)

# Pivot region data to get region mix percentages per week
region_totals = df_region.groupby(['fiscal_year', 'fiscal_week_number'])['total_items_sold'].sum().reset_index()
region_totals.columns = ['fiscal_year', 'fiscal_week_number', 'week_total']

df_region = df_region.merge(region_totals, on=['fiscal_year', 'fiscal_week_number'])
df_region['region_pct'] = 100.0 * df_region['total_items_sold'] / df_region['week_total']

# Pivot to wide format
region_pivot = df_region.pivot_table(
    index=['fiscal_year', 'fiscal_week_number'],
    columns='item_region_id',
    values=['region_pct', 'avg_lot_value'],
    aggfunc='first'
).reset_index()

# Flatten column names
region_pivot.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in region_pivot.columns]

# Merge with base data
df = df_base.merge(region_pivot, on=['fiscal_year', 'fiscal_week_number'], how='left')

print(f"Loaded {len(df)} weeks of data")
print(f"Columns: {list(df.columns)}")

# Create target variable
df['target'] = (df['goal_status'] == 'hit').astype(int)

# Fill NaN values
df = df.fillna(0)

# Get region columns
region_pct_cols = [c for c in df.columns if c.startswith('region_pct_')]
region_alv_cols = [c for c in df.columns if c.startswith('avg_lot_value_')]

print(f"\nRegion columns found: {region_pct_cols}")

# Features: industry + geography
features_industry = [
    'construction_pct', 'construction_alv',
    'ag_pct', 'ag_alv',
    'passenger_pct', 'passenger_alv',
    'trucks_med_heavy_pct', 'trucks_med_heavy_alv',
    'total_items_sold'
]

features_all = features_industry + region_pct_cols + region_alv_cols

X_industry = df[features_industry]
X_all = df[features_all]
y = df['target']

print(f"\nTarget distribution: {y.sum()} hits, {len(y) - y.sum()} misses")

# ============================================================
# COMPARE: INDUSTRY ONLY vs INDUSTRY + GEOGRAPHY
# ============================================================
print("\n" + "="*60)
print("COMPARISON: INDUSTRY ONLY vs INDUSTRY + GEOGRAPHY")
print("="*60)

# Industry only
scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X_industry)
lr1 = LogisticRegression(max_iter=1000, random_state=42)
lr1_cv = cross_val_score(lr1, X1_scaled, y, cv=5)
print(f"\nIndustry features only: {lr1_cv.mean():.1%} (+/- {lr1_cv.std()*2:.1%})")

# Industry + Geography
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X_all)
lr2 = LogisticRegression(max_iter=1000, random_state=42)
lr2_cv = cross_val_score(lr2, X2_scaled, y, cv=5)
print(f"Industry + Geography:   {lr2_cv.mean():.1%} (+/- {lr2_cv.std()*2:.1%})")

improvement = lr2_cv.mean() - lr1_cv.mean()
print(f"\nImprovement from adding geography: {improvement:+.1%}")

# ============================================================
# FEATURE IMPORTANCE WITH GEOGRAPHY
# ============================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE (ALL FEATURES)")
print("="*60)

lr2.fit(X2_scaled, y)

coef_df = pd.DataFrame({
    'feature': features_all,
    'coefficient': lr2.coef_[0],
    'abs_coef': np.abs(lr2.coef_[0])
}).sort_values('abs_coef', ascending=False)

print("\nTop 15 Features by Importance:")
for i, row in coef_df.head(15).iterrows():
    direction = "↑" if row['coefficient'] > 0 else "↓"
    print(f"  {direction} {row['feature']}: {row['coefficient']:.3f}")

# ============================================================
# DECISION TREE WITH GEOGRAPHY
# ============================================================
print("\n" + "="*60)
print("DECISION TREE (WITH GEOGRAPHY)")
print("="*60)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_cv = cross_val_score(dt, X_all, y, cv=5)
print(f"\nCross-validation accuracy: {dt_cv.mean():.1%} (+/- {dt_cv.std()*2:.1%})")

dt.fit(X_all, y)

print("\nFeature Importance from Tree:")
importance_df = pd.DataFrame({
    'feature': features_all,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance_df.head(10).iterrows():
    if row['importance'] > 0.01:
        print(f"  {row['feature']}: {row['importance']:.1%}")

print("\n" + "-"*40)
print("Decision Tree Rules:")
print("-"*40)
print(export_text(dt, feature_names=features_all, max_depth=3))
