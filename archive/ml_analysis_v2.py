import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from src.database import AuctionDatabase

# Connect to database
db = AuctionDatabase()

# Pull enriched data
print("Fetching data...")
data = db.query("""
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

db.close()

# Convert to DataFrame
df = pd.DataFrame(data)
print(f"Loaded {len(df)} weeks of data")

# Create target variable (1 = hit, 0 = miss)
df['target'] = (df['goal_status'] == 'hit').astype(int)

# Fill NaN ALV values with 0 (weeks where that industry had no items)
alv_cols = ['construction_alv', 'ag_alv', 'passenger_alv', 'trucks_med_heavy_alv']
df[alv_cols] = df[alv_cols].fillna(0)

# ACTIONABLE features only - exclude pct_items_10k_plus (that's an outcome, not a lever)
features = [
    'construction_pct', 'construction_alv',
    'ag_pct', 'ag_alv',
    'passenger_pct', 'passenger_alv',
    'trucks_med_heavy_pct', 'trucks_med_heavy_alv',
    'total_items_sold'
]

X = df[features]
y = df['target']

print(f"\nTarget distribution: {y.sum()} hits, {len(y) - y.sum()} misses")
print("\n*** EXCLUDING pct_items_10k_plus (outcome metric) ***")
print("*** FOCUSING ON ACTIONABLE LEVERS ***\n")

# ============================================================
# LOGISTIC REGRESSION
# ============================================================
print("="*60)
print("LOGISTIC REGRESSION (Actionable Features Only)")
print("="*60)

# Scale features for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)

# Cross-validation accuracy
lr_cv_scores = cross_val_score(lr, X_scaled, y, cv=5)
print(f"\nCross-validation accuracy: {lr_cv_scores.mean():.1%} (+/- {lr_cv_scores.std()*2:.1%})")

# Feature importance (coefficients)
print("\nFeature Importance (what actually drives success):")
coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': lr.coef_[0],
    'abs_coef': np.abs(lr.coef_[0])
}).sort_values('abs_coef', ascending=False)

for _, row in coef_df.iterrows():
    direction = "↑ MORE" if row['coefficient'] > 0 else "↓ LESS"
    print(f"  {direction} {row['feature']}: {row['coefficient']:.3f}")

# ============================================================
# DECISION TREE
# ============================================================
print("\n" + "="*60)
print("DECISION TREE (Actionable Features Only)")
print("="*60)

# Try different max_depth to avoid overfitting
for depth in [2, 3, 4]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_cv_scores = cross_val_score(dt, X, y, cv=5)
    print(f"\nDepth {depth} - CV accuracy: {dt_cv_scores.mean():.1%} (+/- {dt_cv_scores.std()*2:.1%})")

# Use depth=3 for interpretability
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

print("\n" + "-"*40)
print("THE RECIPE (Decision Tree Rules):")
print("-"*40)
print(export_text(dt, feature_names=features))

# Feature importance from tree
print("\nActionable Drivers Ranked:")
importance_df = pd.DataFrame({
    'feature': features,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance_df.iterrows():
    if row['importance'] > 0.01:
        print(f"  {row['feature']}: {row['importance']:.1%}")

# ============================================================
# PREDICTIONS
# ============================================================
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

df['dt_pred'] = dt.predict(X)
dt_acc = (df['dt_pred'] == df['target']).mean()

print(f"\nDecision Tree Accuracy: {dt_acc:.1%}")
print(f"\nConfusion Matrix:")
print(f"  True Positives (correctly predicted hits):  {((df['dt_pred']==1) & (df['target']==1)).sum()}")
print(f"  False Positives (predicted hit, was miss): {((df['dt_pred']==1) & (df['target']==0)).sum()}")
print(f"  False Negatives (predicted miss, was hit): {((df['dt_pred']==0) & (df['target']==1)).sum()}")
print(f"  True Negatives (correctly predicted miss):  {((df['dt_pred']==0) & (df['target']==0)).sum()}")

# ============================================================
# BUSINESS INTERPRETATION
# ============================================================
print("\n" + "="*60)
print("BUSINESS INTERPRETATION")
print("="*60)

# Calculate average values for hit vs miss weeks
print("\nAverage Values: HIT weeks vs MISS weeks")
print("-"*50)
for feat in features:
    hit_avg = df[df['target']==1][feat].mean()
    miss_avg = df[df['target']==0][feat].mean()
    diff = hit_avg - miss_avg
    direction = "↑" if diff > 0 else "↓"
    print(f"  {feat}:")
    print(f"    HIT weeks:  {hit_avg:.1f}")
    print(f"    MISS weeks: {miss_avg:.1f}")
    print(f"    Difference: {direction} {abs(diff):.1f}")
    print()
