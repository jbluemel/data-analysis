"""
ML Scorecard Builder for Purple Wave ALV Goal Analysis
======================================================
Trains a Random Forest classifier to identify which metrics drive
hitting or missing the $10k weekly Average Lot Value (ALV) goal.

Usage: python build_and_test_ml_scorecard.py
"""

import json
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib


def connect_to_database():
    """Connect to local PostgreSQL database."""
    conn = psycopg2.connect(
        host="localhost",
        port=5434,
        database="dbt_dev",
        user="dbt_user",
        password="dbt_password"
    )
    return conn


def load_weekly_metrics(conn):
    """Load weekly metrics from the enriched summary table."""
    query = """
    SELECT 
        fiscal_year,
        fiscal_week_number,
        total_items_sold,
        avg_lot_value,
        total_contract_price,
        pct_items_10k_plus,
        pct_items_under_500,
        construction_pct,
        construction_alv,
        passenger_pct,
        passenger_alv,
        trucks_med_heavy_pct,
        trucks_med_heavy_alv,
        ag_pct,
        ag_alv,
        region_3_pct,
        region_3_alv,
        region_4_pct,
        region_4_alv,
        region_5_pct,
        region_5_alv,
        top_revenue_pct,
        top_revenue_alv,
        alv_variance,
        alv_variance_pct,
        goal_status
    FROM weekly_metrics_summary_enrichedv2
    WHERE goal_status IS NOT NULL
      AND has_auctions = true
    ORDER BY fiscal_year, fiscal_week_number
    """
    df = pd.read_sql(query, conn)
    return df


def prepare_features(df):
    """Prepare feature matrix and target variable."""
    feature_columns = [
        'total_items_sold',
        'pct_items_10k_plus',
        'pct_items_under_500',
        'construction_pct',
        'construction_alv',
        'passenger_pct',
        'passenger_alv',
        'trucks_med_heavy_pct',
        'trucks_med_heavy_alv',
        'ag_pct',
        'ag_alv',
        'region_3_pct',
        'region_3_alv',
        'region_4_pct',
        'region_4_alv',
        'region_5_pct',
        'region_5_alv',
        'top_revenue_pct',
        'top_revenue_alv'
    ]
    
    X = df[feature_columns].copy()
    X = X.fillna(X.median())
    
    le = LabelEncoder()
    y = le.fit_transform(df['goal_status'])
    class_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    
    return X, y, feature_columns, class_mapping


def train_and_evaluate_model(X, y, feature_columns):
    """Train Random Forest with cross-validation and return results."""
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
    cv_predictions = cross_val_predict(rf_model, X, y, cv=5)
    rf_model.fit(X, y)
    
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return rf_model, cv_scores, cv_predictions, importance_df


def print_scorecard(importance_df, cv_scores, cv_predictions, y, class_mapping):
    """Print the ML scorecard results."""
    print("\n" + "="*70)
    print("  ML SCORECARD: What Drives Hitting the $10k ALV Goal?")
    print("="*70)
    
    print("\nMODEL PERFORMANCE (5-Fold Cross-Validation)")
    print("-" * 50)
    print(f"  Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
    print(f"  Individual folds: {', '.join([f'{s:.1%}' for s in cv_scores])}")
    
    print("\nFEATURE IMPORTANCE RANKING")
    print("-" * 50)
    print(f"  {'Rank':<6}{'Feature':<30}{'Importance':<15}{'Cumulative'}")
    print("  " + "-" * 60)
    
    cumulative = 0
    for i, (_, row) in enumerate(importance_df.iterrows(), 1):
        cumulative += row['importance']
        bar = "#" * int(row['importance'] * 40)
        print(f"  {i:<6}{row['feature']:<30}{row['importance']:>6.1%}      {cumulative:>6.1%}")
        if i <= 10:
            print(f"        {bar}")
    
    print("\nTOP 5 DRIVERS OF ALV GOAL SUCCESS")
    print("-" * 50)
    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.1%} importance")
    
    print("\nCONFUSION MATRIX (Cross-Validated)")
    print("-" * 50)
    cm = confusion_matrix(y, cv_predictions)
    labels = [class_mapping[i] for i in sorted(class_mapping.keys())]
    
    print(f"\n                    Predicted")
    print(f"                 {labels[0]:>8}  {labels[1]:>8}")
    print(f"  Actual {labels[0]:>8}   {cm[0,0]:>6}    {cm[0,1]:>6}")
    print(f"         {labels[1]:>8}   {cm[1,0]:>6}    {cm[1,1]:>6}")
    
    hit_idx = 0 if labels[0] == 'hit' else 1
    miss_idx = 1 - hit_idx
    
    print(f"\n  Total weeks analyzed: {len(y)}")
    print(f"  Weeks hitting goal: {(y == hit_idx).sum()} ({(y == hit_idx).sum()/len(y):.1%})")
    print(f"  Weeks missing goal: {(y == miss_idx).sum()} ({(y == miss_idx).sum()/len(y):.1%})")
    print("\n" + "="*70)


def save_scorecard_to_json(importance_df, cv_scores, y, class_mapping, filename='ml_scorecard.json'):
    """Save the scorecard results to a JSON file."""
    hit_idx = 0 if list(class_mapping.values())[0] == 'hit' else 1
    miss_idx = 1 - hit_idx
    
    scorecard = {
        'model_info': {
            'algorithm': 'Random Forest',
            'n_estimators': 100,
            'max_depth': 5,
            'cross_validation_folds': 5
        },
        'performance': {
            'mean_accuracy': float(cv_scores.mean()),
            'std_accuracy': float(cv_scores.std()),
            'fold_accuracies': [float(s) for s in cv_scores]
        },
        'feature_importances': [
            {
                'rank': i + 1,
                'feature': row['feature'],
                'importance': float(row['importance'])
            }
            for i, (_, row) in enumerate(importance_df.iterrows())
        ],
        'top_5_drivers': [
            {
                'feature': row['feature'],
                'importance': float(row['importance'])
            }
            for _, row in importance_df.head(5).iterrows()
        ],
        'data_summary': {
            'total_weeks': int(len(y)),
            'weeks_hit': int((y == hit_idx).sum()),
            'weeks_miss': int((y == miss_idx).sum())
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    print(f"\nScorecard saved to: {filename}")


def main():
    """Main execution flow."""
    print("Connecting to PostgreSQL database...")
    conn = connect_to_database()
    
    print("Loading weekly metrics data...")
    df = load_weekly_metrics(conn)
    print(f"   Loaded {len(df)} weeks of data")
    
    conn.close()
    
    print("Preparing features...")
    X, y, feature_columns, class_mapping = prepare_features(df)
    print(f"   Features: {len(feature_columns)}")
    print(f"   Target classes: {list(class_mapping.values())}")
    
    print("Training Random Forest model with 5-fold CV...")
    rf_model, cv_scores, cv_predictions, importance_df = train_and_evaluate_model(
        X, y, feature_columns
    )
    
    print_scorecard(importance_df, cv_scores, cv_predictions, y, class_mapping)
    save_scorecard_to_json(importance_df, cv_scores, y, class_mapping)
    
    joblib.dump(rf_model, 'rf_alv_model.joblib')
    print(f"Model saved to: rf_alv_model.joblib")


if __name__ == "__main__":
    main()