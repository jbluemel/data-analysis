import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from src.database import AuctionDatabase

db = AuctionDatabase()

print("Fetching base data...")
base_data = db.query("""
    SELECT
        fiscal_year,
        fiscal_week_number,
        goal_status,
        construction_pct, construction_alv,
        ag_pct, ag_alv,
        passenger_pct, passenger_alv,
        trucks_med_heavy_pct, trucks_med_heavy_alv,
        total_items_sold
    FROM weekly_metrics_summary_enrichedv2
    WHERE has_auctions = true
""")

print("Fetching region data...")
region_data = db.query("""
    SELECT fiscal_year, fiscal_week_number, item_region_id,
           total_items_sold, avg_lot_value
    FROM weekly_metrics_by_regionv2
""")

print("Fetching district data...")
district_data = db.query("""
    SELECT fiscal_year, fiscal_week_number, item_district,
           total_items_sold, avg_lot_value
    FROM weekly_metrics_by_districtv2
""")

print("Fetching territory data...")
territory_data = db.query("""
    SELECT fiscal_year, fiscal_week_number, item_territory_id,
           total_items_sold, avg_lot_value
    FROM weekly_metrics_by_territoryv2
""")

db.close()

# Convert to DataFrames
df_base = pd.DataFrame(base_data)
df_region = pd.DataFrame(region_data)
df_district = pd.DataFrame(district_data)
df_territory = pd.DataFrame(territory_data)

print(f"\nUnique regions: {df_region['item_region_id'].nunique()}")
print(f"Unique districts: {df_district['item_district'].nunique()}")
print(f"Unique territories: {df_territory['item_territory_id'].nunique()}")

# Function to pivot geographic data
def pivot_geo(df, id_col, prefix):
    totals = df.groupby(['fiscal_year', 'fiscal_week_number'])['total_items_sold'].sum().reset_index()
    totals.columns = ['fiscal_year', 'fiscal_week_number', 'week_total']
    df = df.merge(totals, on=['fiscal_year', 'fiscal_week_number'])
    df['pct'] = 100.0 * df['total_items_sold'] / df['week_total']
    
    pivot = df.pivot_table(
        index=['fiscal_year', 'fiscal_week_number'],
        columns=id_col,
        values=['pct', 'avg_lot_value'],
        aggfunc='first'
    ).reset_index()
    
    pivot.columns = [f'{prefix}_{col[0]}_{col[1]}' if col[1] else col[0] for col in pivot.columns]
    return pivot

# Pivot each geographic level
region_pivot = pivot_geo(df_region, 'item_region_id', 'region')
district_pivot = pivot_geo(df_district, 'item_district', 'district')
territory_pivot = pivot_geo(df_territory, 'item_territory_id', 'territory')

# Merge all
df = df_base.copy()
df = df.merge(region_pivot, on=['fiscal_year', 'fiscal_week_number'], how='left')
df = df.merge(district_pivot, on=['fiscal_year', 'fiscal_week_number'], how='left')
df = df.merge(territory_pivot, on=['fiscal_year', 'fiscal_week_number'], how='left')

df['target'] = (df['goal_status'] == 'hit').astype(int)
df = df.fillna(0)

# Define feature sets
features_industry = [
    'construction_pct', 'construction_alv',
    'ag_pct', 'ag_alv',
    'passenger_pct', 'passenger_alv',
    'trucks_med_heavy_pct', 'trucks_med_heavy_alv',
    'total_items_sold'
]

region_cols = [c for c in df.columns if c.startswith('region_')]
district_cols = [c for c in df.columns if c.startswith('district_')]
territory_cols = [c for c in df.columns if c.startswith('territory_')]

y = df['target']

print(f"\nRegion columns: {len(region_cols)}")
print(f"District columns: {len(district_cols)}")
print(f"Territory columns: {len(territory_cols)}")

# ============================================================
# COMPARE ACCURACY AT EACH LEVEL
# ============================================================
print("\n" + "="*60)
print("ACCURACY COMPARISON BY GEOGRAPHIC LEVEL")
print("="*60)

configs = [
    ("Industry only", features_industry),
    ("Industry + Region", features_industry + region_cols),
    ("Industry + District", features_industry + district_cols),
    ("Industry + Territory", features_industry + territory_cols),
    ("Industry + Region + District", features_industry + region_cols + district_cols),
    ("Industry + All Geography", features_industry + region_cols + district_cols + territory_cols),
]

results = []
for name, features in configs:
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(lr, X_scaled, y, cv=5)
    mean_acc = cv_scores.mean()
    std_acc = cv_scores.std() * 2
    results.append((name, mean_acc, std_acc, len(features)))
    print(f"\n{name}:")
    print(f"  Accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")
    print(f"  Features: {len(features)}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n{'Model':<35} {'Accuracy':<12} {'Features'}")
print("-"*55)
for name, acc, std, n_feat in results:
    print(f"{name:<35} {acc:.1%} +/-{std:.1%}   {n_feat}")
