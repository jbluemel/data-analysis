"""
Purple Wave ALV Goal Scorecard - ML Feature Importance Analysis
===============================================================
Connects to local PostgreSQL, trains a Random Forest classifier to predict
whether weekly Average Lot Value hits the $10,000 target, and outputs a
ranked feature-importance scorecard.

Usage:  python build_and_test_ml_scorecard.py
"""

import json
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# ── 1. Connect & Load ────────────────────────────────────────────────────────

DB_CONFIG = {
    "host": "localhost",
    "port": 5434,
    "database": "dbt_dev",
    "user": "dbt_user",
    "password": "dbt_password",
}

QUERY = """
SELECT *
FROM weekly_metrics_summary_enrichedv2
WHERE goal_status IS NOT NULL
ORDER BY fiscal_year, fiscal_week_number
"""

print("Connecting to PostgreSQL …")
conn = psycopg2.connect(**DB_CONFIG)
df = pd.read_sql(QUERY, conn)
conn.close()
print(f"Loaded {len(df)} weekly records.\n")

# ── 2. Prepare Features & Target ─────────────────────────────────────────────

# Encode target: HIT = 1, MISS = 0
le = LabelEncoder()
df["target"] = le.fit_transform(df["goal_status"].str.upper())
print(f"Class distribution:\n{df['goal_status'].value_counts()}\n")

# Numeric feature columns (exclude identifiers, target-leaking cols, text cols)
EXCLUDE = {
    "fiscal_year",
    "fiscal_week_number",
    "goal_status",
    "target",
    "alv_target",            # constant — no predictive value
    "alv_variance",          # directly derived from ALV (leakage)
    "alv_variance_pct",      # directly derived from ALV (leakage)
    "avg_lot_value",         # this IS the target metric (leakage)
    "total_contract_price",  # ALV * volume (leakage)
    "has_auctions",          # boolean flag, not a driver
    "top_revenue_industry",  # text — handled separately if needed
    "anomaly_industry",      # text — handled separately if needed
}

feature_cols = [
    c for c in df.columns
    if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
]

print(f"Features ({len(feature_cols)}):")
for col in feature_cols:
    print(f"  • {col}")
print()

X = df[feature_cols].copy()
y = df["target"].copy()

# Fill any NaNs with column median (safe for tree models)
X = X.fillna(X.median())

# ── 3. Train Random Forest with 5-Fold CV ────────────────────────────────────

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced",   # handle any HIT/MISS imbalance
)

print("Running 5-fold cross-validation …")
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print(f"  Fold accuracies : {np.round(cv_scores, 4)}")
print(f"  Mean accuracy   : {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})\n")

# Cross-val predictions for confusion matrix
y_pred_cv = cross_val_predict(rf, X, y, cv=5)

# ── 4. Confusion Matrix ──────────────────────────────────────────────────────

cm = confusion_matrix(y, y_pred_cv)
labels = le.classes_   # e.g. ['HIT', 'MISS'] or ['Hit', 'Miss']
print("Confusion Matrix (rows = actual, cols = predicted):")
header = "           " + "  ".join(f"{l:>8}" for l in labels)
print(header)
for i, row_label in enumerate(labels):
    row_vals = "  ".join(f"{cm[i, j]:>8}" for j in range(len(labels)))
    print(f"  {row_label:>8}  {row_vals}")
print()
print(classification_report(y, y_pred_cv, target_names=[str(l) for l in labels]))

# ── 5. Feature Importances (fit on full dataset for final ranking) ────────────

rf.fit(X, y)
importances = rf.feature_importances_
importance_df = (
    pd.DataFrame({"feature": feature_cols, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("=" * 60)
print("  ALV GOAL SCORECARD — Feature Importance Ranking")
print("=" * 60)
for idx, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"  {idx + 1:>2}. {row['feature']:<35} {row['importance']:.4f}  {bar}")
print("=" * 60)
print()

# ── 6. Save Scorecard to JSON ────────────────────────────────────────────────

scorecard = {
    "model": "RandomForest",
    "n_estimators": 100,
    "max_depth": 5,
    "cv_folds": 5,
    "mean_accuracy": round(float(cv_scores.mean()), 4),
    "std_accuracy": round(float(cv_scores.std()), 4),
    "fold_accuracies": [round(float(s), 4) for s in cv_scores],
    "confusion_matrix": {
        "labels": [str(l) for l in labels],
        "matrix": cm.tolist(),
    },
    "feature_importances": [
        {"rank": i + 1, "feature": row["feature"], "importance": round(float(row["importance"]), 4)}
        for i, row in importance_df.iterrows()
    ],
}

OUTPUT_FILE = "ml_scorecard.json"
with open(OUTPUT_FILE, "w") as f:
    json.dump(scorecard, f, indent=2)

print(f"Scorecard saved to {OUTPUT_FILE}")
print("Done.")