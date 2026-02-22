#!/usr/bin/env python3
"""
ALV Scorecard Trainer
Uses Random Forest to identify which metrics drive hitting the $10k weekly ALV goal.
Outputs feature importances and saves the model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import psycopg2

# Database connection settings
DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'dbt_dev',
    'user': 'dbt_user',
    'password': 'dbt_password'
}

# Features to use for prediction
FEATURE_COLUMNS = [
    # Category percentages (mix)
    'construction_pct',
    'passenger_pct', 
    'trucks_med_heavy_pct',
    'ag_pct',
    
    # Category ALVs (value within category)
    'construction_alv',
    'passenger_alv',
    'trucks_med_heavy_alv',
    'ag_alv',
    
    # Regional metrics
    'region_3_pct',
    'region_3_alv',
    'region_4_pct',
    'region_4_alv',
    'region_5_pct',
    'region_5_alv',
    
    # Price distribution
    'pct_items_10k_plus',
    'pct_items_under_500',
    
    # Volume
    'total_items_sold'
]


def load_data():
    """Load weekly metrics from PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = """
    SELECT 
        fiscal_year,
        fiscal_week_number,
        avg_lot_value,
        goal_status,
        construction_pct, COALESCE(construction_alv, 0) as construction_alv,
        passenger_pct, COALESCE(passenger_alv, 0) as passenger_alv,
        trucks_med_heavy_pct, COALESCE(trucks_med_heavy_alv, 0) as trucks_med_heavy_alv,
        ag_pct, COALESCE(ag_alv, 0) as ag_alv,
        region_3_pct, COALESCE(region_3_alv, 0) as region_3_alv,
        region_4_pct, COALESCE(region_4_alv, 0) as region_4_alv,
        region_5_pct, COALESCE(region_5_alv, 0) as region_5_alv,
        pct_items_10k_plus, pct_items_under_500,
        total_items_sold
    FROM weekly_metrics_summary_enrichedv2 
    WHERE has_auctions = true
    ORDER BY fiscal_year, fiscal_week_number
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def prepare_features(df):
    """Prepare feature matrix and target variable."""
    y = (df['goal_status'] == 'hit').astype(int)
    X = df[FEATURE_COLUMNS].copy().fillna(0)
    return X, y


def train_model(X, y):
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def generate_scorecard(model, feature_names):
    """Generate feature importance scorecard."""
    importances = model.feature_importances_
    
    scorecard = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Importance_Pct': importances * 100
    }).sort_values('Importance', ascending=False)
    
    scorecard['Cumulative_Pct'] = scorecard['Importance_Pct'].cumsum()
    scorecard['Rank'] = range(1, len(scorecard) + 1)
    
    return scorecard.reset_index(drop=True)


def print_scorecard(scorecard):
    """Print formatted scorecard to console."""
    print("\n" + "="*70)
    print("ALV GOAL SCORECARD - Random Forest Feature Importances")
    print("="*70)
    print(f"\n{'Rank':<6}{'Feature':<25}{'Importance':<12}{'Cumulative':<12}")
    print("-"*55)
    
    for _, row in scorecard.iterrows():
        print(f"{int(row['Rank']):<6}{row['Feature']:<25}{row['Importance_Pct']:>8.2f}%{row['Cumulative_Pct']:>11.2f}%")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    top_5 = scorecard.head(5)
    print(f"\nTop 5 drivers explain {top_5['Importance_Pct'].sum():.1f}% of goal achievement:")
    for _, row in top_5.iterrows():
        print(f"  • {row['Feature']}: {row['Importance_Pct']:.1f}%")


def main():
    print("Loading data from PostgreSQL (localhost:5434)...")
    df = load_data()
    print(f"Loaded {len(df)} weeks of auction data")
    
    hit_count = (df['goal_status'] == 'hit').sum()
    miss_count = (df['goal_status'] == 'miss').sum()
    print(f"\nClass distribution:")
    print(f"  HITs:   {hit_count} ({hit_count/len(df)*100:.1f}%)")
    print(f"  MISSes: {miss_count} ({miss_count/len(df)*100:.1f}%)")
    
    print("\nPreparing features...")
    X, y = prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    
    print("\nTraining Random Forest model...")
    model = train_model(X, y)
    
    scorecard = generate_scorecard(model, FEATURE_COLUMNS)
    print_scorecard(scorecard)
    
    # Save outputs
    joblib.dump(model, 'alv_rf_model.joblib')
    print(f"\n✓ Model saved to: alv_rf_model.joblib")
    
    scorecard.to_csv('alv_scorecard.csv', index=False)
    print(f"✓ Scorecard saved to: alv_scorecard.csv")
    
    joblib.dump(FEATURE_COLUMNS, 'alv_feature_names.joblib')
    
    train_accuracy = model.score(X, y)
    print(f"\nTraining accuracy: {train_accuracy*100:.1f}%")
    print("(Use test_scorecard.py for proper cross-validation)")
    
    return model, scorecard


if __name__ == '__main__':
    model, scorecard = main()