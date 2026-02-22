#!/usr/bin/env python3
"""
ALV ML Analysis Test
Uses Random Forest to find what really drives hitting the $10k ALV goal
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'dbt_dev',
    'user': 'dbt_user',
    'password': 'dbt_password'
}

def main():
    print("Connecting to database...")
    print(f"  Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"  Database: {DB_CONFIG['database']}")
    print(f"  Table: weekly_metrics_summary_enrichedv2")
    print()
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = """
    SELECT 
        fiscal_year,
        fiscal_week_number,
        avg_lot_value,
        total_items_sold,
        construction_pct,
        passenger_pct,
        trucks_med_heavy_pct,
        ag_pct,
        pct_items_10k_plus,
        pct_items_under_500,
        construction_alv,
        passenger_alv,
        trucks_med_heavy_alv,
        ag_alv,
        region_3_pct,
        region_3_alv,
        region_4_pct,
        region_4_alv,
        region_5_pct,
        region_5_alv,
        goal_status
    FROM weekly_metrics_summary_enrichedv2
    WHERE goal_status IN ('hit', 'miss')
    ORDER BY fiscal_year, fiscal_week_number
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Prepare features
    feature_cols = [
        'construction_pct', 'passenger_pct', 'trucks_med_heavy_pct', 'ag_pct',
        'pct_items_10k_plus', 'pct_items_under_500',
        'construction_alv', 'passenger_alv', 'trucks_med_heavy_alv', 'ag_alv',
        'region_3_pct', 'region_3_alv', 'region_4_pct', 'region_4_alv',
        'region_5_pct', 'region_5_alv', 'total_items_sold'
    ]
    
    X = df[feature_cols].fillna(0)
    y = (df['goal_status'] == 'hit').astype(int)
    
    # Train Random Forest with cross-validation
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    
    # Get predictions using cross-validation (prevents overfitting)
    y_pred = cross_val_predict(rf, X, y, cv=5)
    
    # Fit model on all data for feature importances
    rf.fit(X, y)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total = len(df)
    actual_hits = tp + fn
    actual_misses = fp + tn
    predicted_hits = tp + fp
    predicted_misses = fn + tn
    
    accuracy = accuracy_score(y, y_pred) * 100
    
    # Feature importances
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("=" * 70)
    print("ALV ML ANALYSIS - RANDOM FOREST")
    print("=" * 70)
    print()
    print(f"Dataset: {total} weeks from gold table")
    print(f"Range: {df['fiscal_year'].min()} to {df['fiscal_year'].max()}")
    print()
    print("-" * 70)
    print("APPROACH:")
    print("-" * 70)
    print("  Let machine learning find patterns instead of guessing thresholds")
    print("  Algorithm: Random Forest (100 trees, max depth 5)")
    print("  Validation: 5-fold cross-validation")
    print("-" * 70)
    print()
    print("                         ACTUAL DISTRIBUTION")
    print("-" * 70)
    print(f"Total Weeks: {total}")
    print(f"Actual HITs: {actual_hits} ({actual_hits/total*100:.1f}%)")
    print(f"Actual MISSes: {actual_misses} ({actual_misses/total*100:.1f}%)")
    print()
    print("                           CONFUSION MATRIX")
    print("-" * 70)
    print()
    print("                        ACTUAL")
    print("                  HIT         MISS        Total")
    print("            ┌───────────┬───────────┬───────────┐")
    print(f"       HIT  │    {tp:3d}    │    {fp:3d}    │    {predicted_hits:3d}    │  <- Predicted HIT")
    print(" PREDICTED  ├───────────┼───────────┼───────────┤")
    print(f"       MISS │    {fn:3d}    │    {tn:3d}    │    {predicted_misses:3d}    │  <- Predicted MISS")
    print("            ├───────────┼───────────┼───────────┤")
    print(f"      Total │    {actual_hits:3d}    │    {actual_misses:3d}    │    {total:3d}    │")
    print("            └───────────┴───────────┴───────────┘")
    print()
    print("                           ACCURACY METRICS")
    print("-" * 70)
    print(f"Overall Accuracy:      {accuracy:.2f}%  ({tp + tn}/{total} correct)")
    print()
    print("                        TOP FEATURE IMPORTANCES")
    print("-" * 70)
    for i, row in importances.head(10).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:<25} {row['importance']*100:5.1f}%  {bar}")
    print()
    print("=" * 70)
    print(f"ML achieves {accuracy:.1f}% accuracy on {total} weeks of data.")
    print()
    print(f"Correct: {tp + tn} weeks  |  Incorrect: {fp + fn} weeks")
    print()
    print("KEY DISCOVERY: " + importances.iloc[0]['feature'] + " is the #1 predictor!")
    print("=" * 70)

if __name__ == "__main__":
    main()