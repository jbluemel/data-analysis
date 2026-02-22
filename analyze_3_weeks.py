#!/usr/bin/env python3
"""
Analyze 3 FY26 weeks using DSPy with ML-discovered scorecard

Entry point that uses src/analyzer.py module.
Scorecard context is in analyzer.py - update there when ML changes.

Weeks analyzed:
- Week 9: Worst miss ($5,388)
- Week 4: Barely hit ($10,180)  
- Week 6: Best exceed ($13,021)
"""

import dspy
from src.database import AuctionDatabase
from src.analyzer import AuctionAnalyzer

# ============================================================
# LLM CONFIGURATION
# ============================================================
# Uncomment one:

# Local Llama (free, slower)
# lm = dspy.LM('ollama_chat/llama3.1:8b', api_base='http://localhost:11434')

# Claude API (faster, costs money)
lm = dspy.LM('anthropic/claude-sonnet-4-20250514')

dspy.configure(lm=lm)


# ============================================================
# MAIN
# ============================================================
def main():
    # Connect to database
    db = AuctionDatabase()
    
    # Create analyzer (uses signature from src/analyzer.py)
    analyzer = AuctionAnalyzer(db)
    
    # Weeks to analyze
    weeks = [
        (9, "Worst Miss"),
        (4, "Barely Hit"),
        (6, "Best Exceed")
    ]
    
    print("=" * 70)
    print("FY26 WEEK ANALYSIS - ML-POWERED SCORECARD")
    print("=" * 70)
    
    for week_num, label in weeks:
        # Get ALV for display
        result = db.query(f"""
            SELECT avg_lot_value, goal_status 
            FROM weekly_metrics_summary_enrichedv2
            WHERE fiscal_year = 'FY 2026' AND fiscal_week_number = {week_num}
        """)
        
        if result:
            alv = f"${result[0]['avg_lot_value']:,.0f}"
            status = result[0]['goal_status'].upper()
        else:
            alv = "N/A"
            status = "N/A"
        
        print(f"\n{'='*70}")
        print(f"WEEK {week_num}: {label} | ALV: {alv} | {status}")
        print("=" * 70)
        
        # Call analyzer (scorecard context is in analyzer.py)
        explanation = analyzer.explain_week("FY 2026", week_num)
        print(explanation)
        print()
    
    db.close()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()