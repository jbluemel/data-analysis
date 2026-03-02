"""
NPS Detractor Model — XGBoost + SHAP
======================================
Trains a binary classifier: detractor (1) vs everyone else (0).
Uses behavioral features, topic columns, and recency gaps.
Outputs SHAP feature importance as JSON for dashboard consumption.

Usage:
    python -m analyzers.nps.train_detractor_model
"""

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import shap

from shared.database import AuctionDatabase


# ---- Feature definitions ----
NUMERIC_FEATURES = [
    'bought_item_count',
    'sold_item_count',
    'bought_hammer_total',
    'sold_hammer_total',
    'lifetime_bids',
    'avg_highest_bid',
    'serious_bid_pct',
    'items_bid_on',
    'serious_bids',
    'days_since_last_bid',
    'days_since_last_buy',
    'days_since_last_sell',
    'account_tenure_days',
]

CATEGORICAL_FEATURES = [
    'transaction_role',
    'account_type',
    'bidding_status',
    'engagement_level',
    'customer_lifecycle_stage',
    'bidding_intensity_tier',
    'bidding_intent_tier',
    'avg_bid_value_tier',
]

TOPIC_FEATURES = [
    'topic_fees',
    'topic_item_accuracy',
    'topic_item_condition',
    'topic_auction_mechanics',
    'topic_account',
    'topic_seller_honesty',
    'topic_payment',
    'topic_title_paperwork',
]

FLAG_FEATURES = [
]


def main():
    db = AuctionDatabase()
    print("NPS Detractor Model — XGBoost + SHAP")
    print("=" * 60)

    # ---- Load data ----
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TOPIC_FEATURES + FLAG_FEATURES
    col_str = ', '.join(all_features + ['nps_score_label', 'entity_id'])

    query = f"SELECT {col_str} FROM nps_enriched WHERE entity_id IS NOT NULL"
    rows = db.query(query)
    df = pd.DataFrame(rows, columns=all_features + ['nps_score_label', 'entity_id'])

    print(f"Loaded {len(df)} rows")
    print(f"  Detractors: {(df['nps_score_label'] == 'detractor').sum()}")
    print(f"  Others: {(df['nps_score_label'] != 'detractor').sum()}")

    # ---- Prepare target ----
    df['is_detractor'] = (df['nps_score_label'] == 'detractor').astype(int)

    # ---- Prepare features ----
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype('category').cat.codes.replace(-1, np.nan)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in TOPIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TOPIC_FEATURES + FLAG_FEATURES
    X = df[feature_cols]
    y = df['is_detractor']

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Target: {{0: {(y==0).sum()}, 1: {(y==1).sum()}}}")
    print(f"Detractor rate: {y.mean():.1%}")

    # ---- Train with cross-validation ----
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        random_state=42,
        enable_categorical=False,
    )

    print(f"\nTraining XGBoost (scale_pos_weight={scale_weight:.1f})...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X, y, cv=cv, method='predict')

    # ---- Evaluation ----
    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION RESULTS (5-fold)")
    print(f"{'=' * 60}")
    print(f"\n{classification_report(y, y_pred_cv, target_names=['not_detractor', 'detractor'])}")

    cm = confusion_matrix(y, y_pred_cv)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]:>4}  FP={cm[0][1]:>4}")
    print(f"  FN={cm[1][0]:>4}  TP={cm[1][1]:>4}")

    f1 = f1_score(y, y_pred_cv)
    print(f"\nDetractor F1: {f1:.3f}")

    # ---- Train final model on full data for SHAP ----
    print(f"\nTraining final model on full dataset for SHAP...")
    model.fit(X, y)

    # ---- SHAP Analysis ----
    print(f"\n{'=' * 60}")
    print("SHAP FEATURE IMPORTANCE")
    print(f"{'=' * 60}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP value per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_shap,
    }).sort_values('mean_abs_shap', ascending=False)

    print(f"\n{'FEATURE':<30} {'IMPORTANCE':>10}")
    print("-" * 42)
    for _, row in shap_df.iterrows():
        bar_len = int(row['mean_abs_shap'] / shap_df['mean_abs_shap'].max() * 30)
        bar = "█" * bar_len
        print(f"{row['feature']:<30} {row['mean_abs_shap']:>10.4f}  {bar}")

    # ---- Direction analysis for top features ----
    print(f"\n{'=' * 60}")
    print("SHAP DIRECTION ANALYSIS (top 10 features)")
    print(f"{'=' * 60}")
    print("Positive SHAP = pushes toward detractor")
    print("Negative SHAP = pushes away from detractor\n")

    for _, row in shap_df.head(10).iterrows():
        feat = row['feature']
        feat_idx = feature_cols.index(feat)
        feat_shap = shap_values[:, feat_idx]
        feat_vals = X[feat].values

        # Topic columns: compare -1 vs 0 instead of median split
        if feat.startswith('topic_'):
            neg_mask = feat_vals == -1
            zero_mask = feat_vals == 0

            neg_shap = feat_shap[neg_mask].mean() if neg_mask.sum() > 0 else 0
            zero_shap = feat_shap[zero_mask].mean() if zero_mask.sum() > 0 else 0
            neg_dir = "→ detractor" if neg_shap > 0 else "→ safe"
            zero_dir = "→ detractor" if zero_shap > 0 else "→ safe"

            print(f"  {feat}:")
            print(f"    Complained (-1):     {neg_mask.sum():>5} rows, avg SHAP {neg_shap:+.4f} {neg_dir}")
            print(f"    Not mentioned (0):   {zero_mask.sum():>5} rows, avg SHAP {zero_shap:+.4f} {zero_dir}")
            print()
            continue

        # Numeric features: median split
        valid_mask = ~pd.isna(feat_vals)
        if valid_mask.sum() == 0:
            continue

        valid_vals = feat_vals[valid_mask]
        valid_shap = feat_shap[valid_mask]
        median_val = np.nanmedian(valid_vals)

        high_mask = valid_vals > median_val
        low_mask = valid_vals <= median_val

        high_shap = valid_shap[high_mask].mean() if high_mask.sum() > 0 else 0
        low_shap = valid_shap[low_mask].mean() if low_mask.sum() > 0 else 0

        high_dir = "→ detractor" if high_shap > 0 else "→ safe"
        low_dir = "→ detractor" if low_shap > 0 else "→ safe"

        print(f"  {feat}:")
        print(f"    High values (>{median_val:.0f}): avg SHAP {high_shap:+.4f} {high_dir}")
        print(f"    Low values  (≤{median_val:.0f}): avg SHAP {low_shap:+.4f} {low_dir}")
        print()

    # ---- Build output JSON for dashboard ----
    print(f"\n{'=' * 60}")
    print("SAVING MODEL OUTPUT")
    print(f"{'=' * 60}")

    # Feature importance ranked
    feature_importance = []
    for _, row in shap_df.iterrows():
        feat = row['feature']
        feat_idx = feature_cols.index(feat)
        feat_shap = shap_values[:, feat_idx]
        feat_vals = X[feat].values

        valid_mask = ~pd.isna(feat_vals)
        valid_vals = feat_vals[valid_mask]
        valid_shap = feat_shap[valid_mask]

        # Topic columns: -1 vs 0 analysis
        if feat.startswith('topic_'):
            neg_mask = feat_vals == -1
            zero_mask = feat_vals == 0
            neg_shap_avg = float(feat_shap[neg_mask].mean()) if neg_mask.sum() > 0 else 0
            zero_shap_avg = float(feat_shap[zero_mask].mean()) if zero_mask.sum() > 0 else 0

            feature_importance.append({
                'feature': feat,
                'mean_abs_shap': float(row['mean_abs_shap']),
                'complained_count': int(neg_mask.sum()),
                'complained_shap': round(neg_shap_avg, 4),
                'not_mentioned_shap': round(zero_shap_avg, 4),
                'feature_type': 'topic',
            })
            continue

        # Numeric/categorical features: correlation + median split
        valid_mask = ~pd.isna(feat_vals)
        valid_vals = feat_vals[valid_mask]
        valid_shap = feat_shap[valid_mask]

        if len(valid_vals) > 10:
            corr = float(np.corrcoef(valid_vals, valid_shap)[0, 1])
        else:
            corr = 0.0

        median_val = float(np.nanmedian(valid_vals)) if len(valid_vals) > 0 else 0
        high_mask = valid_vals > median_val
        low_mask = valid_vals <= median_val
        high_shap_avg = float(valid_shap[high_mask].mean()) if high_mask.sum() > 0 else 0
        low_shap_avg = float(valid_shap[low_mask].mean()) if low_mask.sum() > 0 else 0

        feature_importance.append({
            'feature': feat,
            'mean_abs_shap': float(row['mean_abs_shap']),
            'correlation': round(corr, 4),
            'median_value': round(median_val, 2),
            'high_value_shap': round(high_shap_avg, 4),
            'low_value_shap': round(low_shap_avg, 4),
            'feature_type': 'recency' if feat.startswith('days_') else
                           'categorical' if feat in CATEGORICAL_FEATURES else 'behavioral',
        })

    output = {
        'model_type': 'detractor',
        'target': 'is_detractor (1) vs everyone else (0)',
        'total_rows': int(len(df)),
        'detractor_count': int(y.sum()),
        'detractor_rate': round(float(y.mean()), 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': {
            'tn': int(cm[0][0]), 'fp': int(cm[0][1]),
            'fn': int(cm[1][0]), 'tp': int(cm[1][1]),
        },
        'feature_importance': feature_importance,
    }

    output_path = "analyzers/nps/detractor_model_output.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to: {output_path}")

    # Save model
    model.save_model("analyzers/nps/detractor_model.json")
    print(f"Model saved to: analyzers/nps/detractor_model.json")

    print(f"\nDone.")


if __name__ == "__main__":
    main()