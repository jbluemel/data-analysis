"""
ALV Scorecard Accuracy Test
Tests the prediction accuracy of the ALV scorecard against historical data
from the weekly_metrics_summary_enrichedv2 gold table.

Scorecard Rule:
- If Construction % >= 25% AND Passenger % <= 20% → Predict HIT
- Otherwise → Predict MISS
"""

import psycopg2
import pandas as pd

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5434,
    "database": "dbt_dev",
    "user": "dbt_user",
    "password": "dbt_password"
}

# ============================================================================
# SCORECARD THRESHOLDS
# ============================================================================

THRESHOLDS = {
    "construction_pct": {"target": 25, "op": ">="},
    "passenger_pct": {"target": 20, "op": "<="},
    "trucks_pct": {"min": 12, "max": 20},
    "pct_items_10k_plus": {"target": 25, "op": ">="},
    "pct_items_under_500": {"target": 15, "op": "<="}
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def fetch_weekly_data():
    """Fetch all weeks from the gold table."""
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = """
        SELECT 
            fiscal_year,
            fiscal_week_number,
            construction_pct,
            passenger_pct,
            trucks_med_heavy_pct as trucks_pct,
            pct_items_10k_plus,
            pct_items_under_500,
            goal_status
        FROM weekly_metrics_summary_enrichedv2
        ORDER BY fiscal_year, fiscal_week_number
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df


def predict_outcome(row):
    """Apply scorecard rule: Construction >= 25% AND Passenger <= 20% → HIT"""
    if row['construction_pct'] >= 25 and row['passenger_pct'] <= 20:
        return 'hit'
    return 'miss'


def run_accuracy_test():
    """Run the scorecard accuracy test."""
    
    # Fetch data
    print("Connecting to database...")
    print(f"  Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"  Database: {DB_CONFIG['database']}")
    print(f"  Table: weekly_metrics_summary_enrichedv2")
    print()
    
    df = fetch_weekly_data()
    
    print("=" * 70)
    print("ALV SCORECARD ACCURACY TEST")
    print("=" * 70)
    print(f"\nDataset: {len(df)} weeks from gold table")
    print(f"Range: {df['fiscal_year'].min()} to {df['fiscal_year'].max()}")
    
    print("\n" + "-" * 70)
    print("SCORECARD RULE:")
    print("-" * 70)
    print("Predict HIT if: Construction % >= 25% AND Passenger % <= 20%")
    print("Predict MISS otherwise")
    print("-" * 70)
    
    # Apply predictions
    df['prediction'] = df.apply(predict_outcome, axis=1)
    df['correct'] = df['prediction'] == df['goal_status']
    
    # Confusion matrix
    tp = len(df[(df['prediction'] == 'hit') & (df['goal_status'] == 'hit')])
    tn = len(df[(df['prediction'] == 'miss') & (df['goal_status'] == 'miss')])
    fp = len(df[(df['prediction'] == 'hit') & (df['goal_status'] == 'miss')])
    fn = len(df[(df['prediction'] == 'miss') & (df['goal_status'] == 'hit')])
    
    # Metrics
    total = len(df)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Actuals
    actual_hits = len(df[df['goal_status'] == 'hit'])
    actual_misses = len(df[df['goal_status'] == 'miss'])
    
    print(f"\n{'ACTUAL DISTRIBUTION':^70}")
    print("-" * 70)
    print(f"Total Weeks: {total}")
    print(f"Actual HITs: {actual_hits} ({actual_hits/total*100:.1f}%)")
    print(f"Actual MISSes: {actual_misses} ({actual_misses/total*100:.1f}%)")
    
    print(f"\n{'CONFUSION MATRIX':^70}")
    print("-" * 70)
    print(f"""
                        ACTUAL
                  HIT         MISS
            ┌───────────┬───────────┐
       HIT  │    {tp:3d}    │    {fp:3d}    │  <- Predicted HIT
 PREDICTED  ├───────────┼───────────┤
       MISS │    {fn:3d}    │    {tn:3d}    │  <- Predicted MISS
            └───────────┴───────────┘
    """)
    
    print(f"{'INTERPRETATION':^70}")
    print("-" * 70)
    print(f"True Positives (TP):  {tp:3d} - Correctly predicted HIT")
    print(f"True Negatives (TN):  {tn:3d} - Correctly predicted MISS")
    print(f"False Positives (FP): {fp:3d} - Predicted HIT, was actually MISS")
    print(f"False Negatives (FN): {fn:3d} - Predicted MISS, was actually HIT")
    
    print(f"\n{'ACCURACY METRICS':^70}")
    print("-" * 70)
    print(f"Overall Accuracy:     {accuracy*100:6.2f}%  ({tp + tn}/{total} correct)")
    print(f"Precision (HIT):      {precision*100:6.2f}%  (When we predict HIT, how often correct?)")
    print(f"Recall (HIT):         {recall*100:6.2f}%  (What % of actual HITs did we catch?)")
    print(f"F1 Score:             {f1*100:6.2f}%  (Harmonic mean of precision & recall)")
    
    # Error analysis
    print(f"\n{'ERROR ANALYSIS':^70}")
    print("-" * 70)
    
    fp_rows = df[(df['prediction'] == 'hit') & (df['goal_status'] == 'miss')]
    if len(fp_rows) > 0:
        print(f"\nFalse Positives ({len(fp_rows)} weeks) - Predicted HIT, Actual MISS:")
        print("Met the rule but still missed goal:")
        for _, row in fp_rows.iterrows():
            print(f"  {row['fiscal_year']} W{int(row['fiscal_week_number']):02d}: "
                  f"Constr={row['construction_pct']:.1f}%, Pass={row['passenger_pct']:.1f}%, "
                  f"$10k+={row['pct_items_10k_plus']:.1f}%")
    
    fn_rows = df[(df['prediction'] == 'miss') & (df['goal_status'] == 'hit')]
    if len(fn_rows) > 0:
        print(f"\nFalse Negatives ({len(fn_rows)} weeks) - Predicted MISS, Actual HIT:")
        print("Broke the rule but still hit goal:")
        for _, row in fn_rows.iterrows():
            print(f"  {row['fiscal_year']} W{int(row['fiscal_week_number']):02d}: "
                  f"Constr={row['construction_pct']:.1f}%, Pass={row['passenger_pct']:.1f}%, "
                  f"$10k+={row['pct_items_10k_plus']:.1f}%")
    
    # Summary
    print(f"\n{'SUMMARY':^70}")
    print("=" * 70)
    print(f"The scorecard achieves {accuracy*100:.1f}% accuracy on {total} weeks of data.")
    print(f"\nCorrect: {tp + tn} weeks  |  Confused: {fp + fn} weeks")
    print("=" * 70)


if __name__ == "__main__":
    run_accuracy_test()