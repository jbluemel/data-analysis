#!/usr/bin/env python3
"""
ALV Scorecard Test - Version 2 (5 thresholds)
Tests the expanded scorecard against historical data
"""

import psycopg2
import pandas as pd

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
        construction_pct,
        passenger_pct,
        trucks_med_heavy_pct,
        pct_items_10k_plus,
        pct_items_under_500,
        goal_status
    FROM weekly_metrics_summary_enrichedv2
    WHERE goal_status IN ('hit', 'miss')
    ORDER BY fiscal_year, fiscal_week_number
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Apply scorecard rules
    def predict(row):
        construction_met = row['construction_pct'] >= 25
        passenger_met = row['passenger_pct'] <= 20
        trucks_met = 12 <= row['trucks_med_heavy_pct'] <= 20
        items_10k_met = row['pct_items_10k_plus'] >= 25
        items_500_met = row['pct_items_under_500'] <= 15
        
        # Prediction logic: Primary (Construction + Passenger) AND Validation (Items 10k+)
        if construction_met and passenger_met and items_10k_met:
            return 'hit'
        else:
            return 'miss'
    
    df['predicted'] = df.apply(predict, axis=1)
    
    # Calculate confusion matrix
    tp = len(df[(df['goal_status'] == 'hit') & (df['predicted'] == 'hit')])
    tn = len(df[(df['goal_status'] == 'miss') & (df['predicted'] == 'miss')])
    fp = len(df[(df['goal_status'] == 'miss') & (df['predicted'] == 'hit')])
    fn = len(df[(df['goal_status'] == 'hit') & (df['predicted'] == 'miss')])
    
    total = len(df)
    actual_hits = tp + fn
    actual_misses = fp + tn
    predicted_hits = tp + fp
    predicted_misses = fn + tn
    
    accuracy = (tp + tn) / total * 100
    
    print("=" * 70)
    print("ALV SCORECARD ACCURACY TEST - VERSION 2 (5 THRESHOLDS)")
    print("=" * 70)
    print()
    print(f"Dataset: {total} weeks from gold table")
    print(f"Range: {df['fiscal_year'].min()} to {df['fiscal_year'].max()}")
    print()
    print("-" * 70)
    print("SCORECARD RULES:")
    print("-" * 70)
    print("  Construction %  >= 25%  (Primary Driver)")
    print("  Passenger %     <= 20%  (Dilution Risk)")
    print("  Trucks %        12-20%  (Stable Base - monitor)")
    print("  Items $10k+     >= 25%  (Validation)")
    print("  Items <$500     <= 15%  (Monitor)")
    print()
    print("Predict HIT if: Construction >= 25% AND Passenger <= 20% AND Items $10k+ >= 25%")
    print("Predict MISS otherwise")
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
    print("=" * 70)
    print(f"The scorecard achieves {accuracy:.1f}% accuracy on {total} weeks of data.")
    print()
    print(f"Correct: {tp + tn} weeks  |  Incorrect: {fp + fn} weeks")
    print("=" * 70)

if __name__ == "__main__":
    main()