import dspy
import json
import os
from shared.database import AuctionDatabase
from typing import Dict, List, Optional


class ExplainWeekPerformance(dspy.Signature):
    """Explain why a week hit, missed, or exceeded the ALV goal based on industry mix, geography, and quality."""

    week_data = dspy.InputField(desc="Week metrics including industry, region percentages and ALVs with benchmarks")
    framework = dspy.InputField(desc="Explanation framework with driver priorities and benchmarks")
    explanation = dspy.OutputField(desc="5-7 bullet points, each starting with •, explaining key drivers and headwinds")


class AuctionAnalyzer(dspy.Module):
    """Auction data analyzer for explaining weekly ALV performance."""

    def __init__(self, db: AuctionDatabase, scorecard_path: str = "ml_scorecard.json"):
        super().__init__()
        self.db = db
        self.explain = dspy.ChainOfThought(ExplainWeekPerformance)
        
        # Load ML scorecard from JSON
        self.scorecard = self._load_scorecard(scorecard_path)
        
        # Load thresholds from database (HIT vs MISS averages)
        self.thresholds = self._load_thresholds()

    def _load_scorecard(self, path: str) -> dict:
        """Load ML scorecard from JSON file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: {path} not found, using defaults")
            return {
                "cv_mean_accuracy": 0.911,
                "features_ranked": [
                    {"rank": 1, "feature": "region_3_alv", "importance_pct": 21.1},
                    {"rank": 2, "feature": "pct_items_10k_plus", "importance_pct": 15.9},
                    {"rank": 3, "feature": "construction_alv", "importance_pct": 13.4},
                    {"rank": 4, "feature": "region_4_alv", "importance_pct": 8.8},
                    {"rank": 5, "feature": "passenger_pct", "importance_pct": 7.3},
                ]
            }

    def _load_thresholds(self) -> dict:
        """Query database for HIT vs MISS averages to use as thresholds."""
        results = self.db.query("""
            SELECT 
                goal_status,
                ROUND(AVG(region_3_alv)::numeric, 0) as avg_region_3_alv,
                ROUND(AVG(region_4_alv)::numeric, 0) as avg_region_4_alv,
                ROUND(AVG(construction_alv)::numeric, 0) as avg_construction_alv,
                ROUND(AVG(construction_pct)::numeric, 1) as avg_construction_pct,
                ROUND(AVG(pct_items_10k_plus)::numeric, 1) as avg_pct_10k_plus,
                ROUND(AVG(passenger_pct)::numeric, 1) as avg_passenger_pct
            FROM weekly_metrics_summary_enrichedv2
            WHERE goal_status IN ('hit', 'miss')
            GROUP BY goal_status
        """)
        
        thresholds = {}
        for row in results:
            thresholds[row['goal_status']] = row
        
        return thresholds

    def _get_top_features(self, n: int = 5) -> List[dict]:
        """Get top N features from scorecard."""
        return self.scorecard.get('features_ranked', [])[:n]

    def _get_accuracy(self) -> float:
        """Get model accuracy from scorecard."""
        return self.scorecard.get('cv_mean_accuracy', 0.911)

    def _get_importance(self, feature: str) -> float:
        """Get importance percentage for a specific feature."""
        for f in self.scorecard.get('features_ranked', []):
            if f['feature'] == feature:
                return f['importance_pct']
        return 0.0

    def query(self, sql: str) -> List[Dict]:
        """Execute a query and return results."""
        return self.db.query(sql)

    def _fmt_dollar(self, val):
        if val is None:
            return "N/A"
        return f"${val:,.0f}"

    def _fmt_pct(self, val):
        if val is None:
            return "N/A"
        return f"{val:.1f}%"

    def _calc_threshold(self, hit_val, miss_val) -> float:
        """Calculate threshold as midpoint between HIT and MISS averages."""
        return (float(hit_val) + float(miss_val)) / 2

    def explain_week(self, fiscal_year: str, fiscal_week: int) -> str:
        """Explain why a specific week hit, missed, or exceeded the ALV goal."""

        print(f"Explaining {fiscal_year} Week {fiscal_week}...")

        # Query enriched summary
        results = self.query(f"""
            SELECT
                fiscal_year,
                fiscal_week_number,
                total_items_sold,
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
                pct_items_10k_plus,
                region_3_pct,
                region_3_alv,
                region_4_pct,
                region_4_alv,
                region_5_pct,
                region_5_alv,
                alv_target,
                alv_variance,
                alv_variance_pct,
                top_revenue_industry,
                top_revenue_pct,
                top_revenue_alv,
                anomaly_industry,
                anomaly_alv,
                anomaly_pct_above_typical
            FROM weekly_metrics_summary_enrichedv2
            WHERE fiscal_year = '{fiscal_year}' AND fiscal_week_number = {fiscal_week}
        """)

        if not results:
            return f"No data found for {fiscal_year} Week {fiscal_week}"

        week = results[0]
        hit = self.thresholds.get('hit', {})
        miss = self.thresholds.get('miss', {})

        # Determine outcome category
        alv = week['avg_lot_value'] or 0
        if alv >= 12000:
            outcome = "EXCEEDED"
        elif alv >= 10000:
            outcome = "HIT"
        else:
            outcome = "MISSED"

        # Get accuracy and top features from loaded scorecard
        accuracy = self._get_accuracy() * 100
        region_3_imp = self._get_importance('region_3_alv')
        items_10k_imp = self._get_importance('pct_items_10k_plus')
        construction_imp = self._get_importance('construction_alv')
        region_4_imp = self._get_importance('region_4_alv')
        passenger_imp = self._get_importance('passenger_pct')

        # Calculate thresholds from HIT/MISS averages
        region_3_threshold = self._calc_threshold(hit.get('avg_region_3_alv', 10474), miss.get('avg_region_3_alv', 7031))
        region_4_threshold = self._calc_threshold(hit.get('avg_region_4_alv', 14045), miss.get('avg_region_4_alv', 10018))
        construction_threshold = self._calc_threshold(hit.get('avg_construction_alv', 19794), miss.get('avg_construction_alv', 12274))
        items_10k_threshold = self._calc_threshold(hit.get('avg_pct_10k_plus', 28.8), miss.get('avg_pct_10k_plus', 22.2))
        passenger_threshold = 20  # Lower is better, use fixed threshold

        # Build week data with ML-discovered benchmarks
        week_data = f"""
WEEK: {fiscal_year} Week {int(week['fiscal_week_number'])}
OUTCOME: {outcome} - ALV was {self._fmt_dollar(week['avg_lot_value'])} vs $10,000 target ({self._fmt_dollar(week['alv_variance'])} variance)
ITEMS SOLD: {int(week['total_items_sold']) if week['total_items_sold'] else 0:,}

TOP REVENUE CONTRIBUTOR:
  Industry: {week['top_revenue_industry']}
  Revenue Share: {self._fmt_pct(week['top_revenue_pct'])}
  ALV: {self._fmt_dollar(week['top_revenue_alv'])}

============================================================
ML SCORECARD - TOP 5 DRIVERS ({accuracy:.1f}% accuracy)
============================================================

#1 REGION 3 ALV ({region_3_imp:.1f}% importance)
  This Week: {self._fmt_dollar(week['region_3_alv'])}
  HIT Avg: {self._fmt_dollar(hit.get('avg_region_3_alv'))} | MISS Avg: {self._fmt_dollar(miss.get('avg_region_3_alv'))} | Threshold: ~{self._fmt_dollar(region_3_threshold)}
  Mix: {self._fmt_pct(week['region_3_pct'])}

#2 ITEMS $10K+ PERCENTAGE ({items_10k_imp:.1f}% importance)
  This Week: {self._fmt_pct(week['pct_items_10k_plus'])}
  HIT Avg: {hit.get('avg_pct_10k_plus')}% | MISS Avg: {miss.get('avg_pct_10k_plus')}% | Threshold: ~{items_10k_threshold:.0f}%

#3 CONSTRUCTION ALV ({construction_imp:.1f}% importance)
  This Week: {self._fmt_dollar(week['construction_alv'])}
  HIT Avg: {self._fmt_dollar(hit.get('avg_construction_alv'))} | MISS Avg: {self._fmt_dollar(miss.get('avg_construction_alv'))} | Threshold: ~{self._fmt_dollar(construction_threshold)}
  Mix: {self._fmt_pct(week['construction_pct'])} (HIT Avg: {hit.get('avg_construction_pct')}% | MISS Avg: {miss.get('avg_construction_pct')}%)

#4 REGION 4 ALV ({region_4_imp:.1f}% importance)
  This Week: {self._fmt_dollar(week['region_4_alv'])}
  HIT Avg: {self._fmt_dollar(hit.get('avg_region_4_alv'))} | MISS Avg: {self._fmt_dollar(miss.get('avg_region_4_alv'))} | Threshold: ~{self._fmt_dollar(region_4_threshold)}
  Mix: {self._fmt_pct(week['region_4_pct'])}

#5 PASSENGER PERCENTAGE ({passenger_imp:.1f}% importance - DILUTION RISK)
  This Week: {self._fmt_pct(week['passenger_pct'])}
  HIT Avg: {hit.get('avg_passenger_pct')}% | MISS Avg: {miss.get('avg_passenger_pct')}% | Threshold: <{passenger_threshold}% (lower is better)
  ALV: {self._fmt_dollar(week['passenger_alv'])}

============================================================
OTHER METRICS
============================================================

TRUCKS, MEDIUM AND HEAVY DUTY
  ALV: {self._fmt_dollar(week['trucks_med_heavy_alv'])}
  Mix: {self._fmt_pct(week['trucks_med_heavy_pct'])}
  Note: Stable base (~15%), not a driver of HIT vs MISS

AG EQUIPMENT
  ALV: {self._fmt_dollar(week['ag_alv'])}
  Mix: {self._fmt_pct(week['ag_pct'])}

REGION 5
  ALV: {self._fmt_dollar(week['region_5_alv'])}
  Mix: {self._fmt_pct(week['region_5_pct'])}
"""

        # Add anomaly section if present
        if week['anomaly_industry']:
            week_data += f"""
ANOMALY DETECTED:
  Industry: {week['anomaly_industry']}
  ALV: {self._fmt_dollar(week['anomaly_alv'])} ({week['anomaly_pct_above_typical']}% above typical)
"""

        # Build framework with loaded scorecard values
        framework = f"""
EXPLANATION FRAMEWORK:

You are explaining to leadership WHY this week's ALV turned out the way it did.

ML SCORECARD CONTEXT:
Random Forest model with {accuracy:.1f}% accuracy identified these top 5 drivers:
1. Region 3 ALV ({region_3_imp:.1f}%) - #1 predictor
2. Items $10k+ % ({items_10k_imp:.1f}%) - high-value inventory concentration
3. Construction ALV ({construction_imp:.1f}%) - quality of construction equipment
4. Region 4 ALV ({region_4_imp:.1f}%) - secondary regional quality
5. Passenger % ({passenger_imp:.1f}%) - dilution risk (lower is better)

STRUCTURE YOUR EXPLANATION:
1. Lead with the TOP driver that explains this week (usually Region 3 ALV)
2. Mention 2-3 supporting factors from the top 5
3. Note any anomalies if present
4. Keep it to 2-3 paragraphs

THRESHOLDS FOR HIT:
- Region 3 ALV > {self._fmt_dollar(region_3_threshold)}
- Items $10k+ > {items_10k_threshold:.0f}%
- Construction ALV > {self._fmt_dollar(construction_threshold)}
- Region 4 ALV > {self._fmt_dollar(region_4_threshold)}
- Passenger % < {passenger_threshold}%

HOW TO EXPLAIN:

For HIT/EXCEEDED weeks:
- Which drivers were above threshold?
- What combination made it work?

For MISSED weeks:
- Which drivers were below threshold?
- What was the primary drag?

TONE:
- Be direct and specific with numbers
- Compare actual values to thresholds
- 2-3 paragraphs maximum
- Use the driver rankings to prioritize what you mention
"""

        result = self.explain(week_data=week_data, framework=framework)
        return result.explanation