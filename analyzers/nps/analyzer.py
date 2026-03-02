import dspy
import json
from shared.database import AuctionDatabase
from typing import Dict, List


class DescribeNPSPerformance(dspy.Signature):
    """Analyze NPS survey data and identify the most important patterns and trends."""

    nps_data = dspy.InputField(desc="Comprehensive NPS metrics including overall scores, trends, and breakdowns by customer segment")
    framework = dspy.InputField(desc="Analysis framework with key dimensions and what to look for")
    analysis = dspy.OutputField(desc="5-7 key insights, each starting with •, backed by specific numbers. Lead with the most actionable finding.")


class NPSAnalyzer(dspy.Module):
    """NPS data analyzer for describing survey performance across customer segments."""

    def __init__(self, db: AuctionDatabase):
        super().__init__()
        self.db = db
        self.describe = dspy.ChainOfThought(DescribeNPSPerformance)

    def query(self, sql: str) -> List[Dict]:
        """Execute a query and return results."""
        return self.db.query(sql)

    def _fmt_pct(self, val):
        if val is None:
            return "N/A"
        return f"{val:.1f}%"

    def _fmt_dollar(self, val):
        if val is None:
            return "N/A"
        return f"${val:,.0f}"

    def _calc_nps(self, promoter_pct, detractor_pct):
        """Calculate NPS score from promoter and detractor percentages."""
        if promoter_pct is None or detractor_pct is None:
            return "N/A"
        return f"{promoter_pct - detractor_pct:.1f}"

    # =========================================================================
    # DATA GATHERING METHODS
    # =========================================================================

    def _get_overall_summary(self) -> str:
        """Get high-level NPS metrics."""
        results = self.query("""
            SELECT
                count(*) as total_responses,
                round(avg(nps_score), 1) as avg_score,
                min(nps_submitted_date) as earliest,
                max(nps_submitted_date) as latest,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'passive' then 1 end) / count(*), 1) as passive_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
        """)
        r = results[0]
        nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
        return f"""OVERALL NPS SUMMARY
  Total Responses: {r['total_responses']:,}
  Date Range: {str(r['earliest'])[:10]} to {str(r['latest'])[:10]}
  Average Score: {r['avg_score']}
  NPS Score: {nps} (Promoters {self._fmt_pct(r['promoter_pct'])} - Detractors {self._fmt_pct(r['detractor_pct'])})
  Promoters: {self._fmt_pct(r['promoter_pct'])} | Passives: {self._fmt_pct(r['passive_pct'])} | Detractors: {self._fmt_pct(r['detractor_pct'])}
"""

    def _get_quarterly_trend(self) -> str:
        """Get NPS trend by fiscal quarter."""
        results = self.query("""
            SELECT
                nps_submitted_fiscal_year as fy,
                nps_submitted_fiscal_quarter as fq,
                count(*) as responses,
                round(avg(nps_score), 1) as avg_score,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
            GROUP BY nps_submitted_fiscal_year, nps_submitted_fiscal_quarter
            ORDER BY nps_submitted_fiscal_year, nps_submitted_fiscal_quarter
        """)
        lines = ["QUARTERLY TREND"]
        for r in results:
            nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
            lines.append(f"  FY{r['fy']} Q{r['fq']}: Avg {r['avg_score']} | NPS {nps} | Promoters {self._fmt_pct(r['promoter_pct'])} | Detractors {self._fmt_pct(r['detractor_pct'])} | n={r['responses']}")
        return "\n".join(lines)

    def _get_by_transaction_role(self) -> str:
        """Get NPS by buyer/seller/both/never."""
        results = self.query("""
            SELECT
                transaction_role,
                count(*) as cnt,
                round(avg(nps_score), 1) as avg_score,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
            GROUP BY transaction_role
            ORDER BY avg_score DESC
        """)
        lines = ["BY TRANSACTION ROLE"]
        for r in results:
            nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
            lines.append(f"  {r['transaction_role']}: Avg {r['avg_score']} | NPS {nps} | Detractors {self._fmt_pct(r['detractor_pct'])} | n={r['cnt']}")
        return "\n".join(lines)

    def _get_by_bidding_intensity(self) -> str:
        """Get NPS by bidding intensity tier."""
        results = self.query("""
            SELECT
                bidding_intensity_tier,
                count(*) as cnt,
                round(avg(nps_score), 1) as avg_score,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
            GROUP BY bidding_intensity_tier
            ORDER BY avg_score DESC
        """)
        lines = ["BY BIDDING INTENSITY"]
        for r in results:
            nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
            lines.append(f"  {r['bidding_intensity_tier']}: Avg {r['avg_score']} | NPS {nps} | Detractors {self._fmt_pct(r['detractor_pct'])} | n={r['cnt']}")
        return "\n".join(lines)

    def _get_by_win_rate(self) -> str:
        """Get NPS by win rate tier."""
        results = self.query("""
            SELECT
                win_rate_tier,
                count(*) as cnt,
                round(avg(nps_score), 1) as avg_score,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
            GROUP BY win_rate_tier
            ORDER BY avg_score DESC
        """)
        lines = ["BY WIN RATE"]
        for r in results:
            nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
            lines.append(f"  {r['win_rate_tier']}: Avg {r['avg_score']} | NPS {nps} | Detractors {self._fmt_pct(r['detractor_pct'])} | n={r['cnt']}")
        return "\n".join(lines)

    def _get_by_engagement_level(self) -> str:
        """Get NPS by engagement level."""
        results = self.query("""
            SELECT
                engagement_level,
                count(*) as cnt,
                round(avg(nps_score), 1) as avg_score,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
            GROUP BY engagement_level
            ORDER BY avg_score DESC
        """)
        lines = ["BY ENGAGEMENT LEVEL"]
        for r in results:
            nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
            lines.append(f"  {r['engagement_level']}: Avg {r['avg_score']} | NPS {nps} | Detractors {self._fmt_pct(r['detractor_pct'])} | n={r['cnt']}")
        return "\n".join(lines)

    def _get_by_account_tenure(self) -> str:
        """Get NPS by account tenure tier."""
        results = self.query("""
            SELECT
                account_tenure_tier,
                count(*) as cnt,
                round(avg(nps_score), 1) as avg_score,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
            GROUP BY account_tenure_tier
            ORDER BY avg_score DESC
        """)
        lines = ["BY ACCOUNT TENURE"]
        for r in results:
            nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
            lines.append(f"  {r['account_tenure_tier']}: Avg {r['avg_score']} | NPS {nps} | Detractors {self._fmt_pct(r['detractor_pct'])} | n={r['cnt']}")
        return "\n".join(lines)

    def _get_by_region(self) -> str:
        """Get NPS by region."""
        results = self.query("""
            SELECT
                coalesce(region_name, 'Unknown') as region_name,
                count(*) as cnt,
                round(avg(nps_score), 1) as avg_score,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
            GROUP BY region_name
            HAVING count(*) >= 20
            ORDER BY avg_score DESC
        """)
        lines = ["BY REGION (min 20 responses)"]
        for r in results:
            nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
            lines.append(f"  {r['region_name']}: Avg {r['avg_score']} | NPS {nps} | Detractors {self._fmt_pct(r['detractor_pct'])} | n={r['cnt']}")
        return "\n".join(lines)

    def _get_by_bid_value_tier(self) -> str:
        """Get NPS by average bid value tier."""
        results = self.query("""
            SELECT
                avg_bid_value_tier,
                count(*) as cnt,
                round(avg(nps_score), 1) as avg_score,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
            GROUP BY avg_bid_value_tier
            ORDER BY avg_score DESC
        """)
        lines = ["BY AVERAGE BID VALUE"]
        for r in results:
            nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
            lines.append(f"  {r['avg_bid_value_tier']}: Avg {r['avg_score']} | NPS {nps} | Detractors {self._fmt_pct(r['detractor_pct'])} | n={r['cnt']}")
        return "\n".join(lines)

    def _get_by_bid_recency(self) -> str:
        """Get NPS by bid recency relative to survey."""
        results = self.query("""
            SELECT
                bid_recency,
                count(*) as cnt,
                round(avg(nps_score), 1) as avg_score,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
            GROUP BY bid_recency
            ORDER BY avg_score DESC
        """)
        lines = ["BY BID RECENCY (relative to survey date)"]
        for r in results:
            nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
            lines.append(f"  {r['bid_recency']}: Avg {r['avg_score']} | NPS {nps} | Detractors {self._fmt_pct(r['detractor_pct'])} | n={r['cnt']}")
        return "\n".join(lines)

    def _get_copart_comparison(self) -> str:
        """Get NPS comparison between Copart and non-Copart users."""
        results = self.query("""
            SELECT
                case when copart_user = true then 'Copart User' else 'Non-Copart User' end as user_type,
                count(*) as cnt,
                round(avg(nps_score), 1) as avg_score,
                round(100.0 * count(case when nps_score_label = 'promoter' then 1 end) / count(*), 1) as promoter_pct,
                round(100.0 * count(case when nps_score_label = 'detractor' then 1 end) / count(*), 1) as detractor_pct
            FROM nps_enriched
            GROUP BY copart_user
            ORDER BY avg_score DESC
        """)
        lines = ["COPART vs NON-COPART USERS"]
        for r in results:
            nps = self._calc_nps(r['promoter_pct'], r['detractor_pct'])
            lines.append(f"  {r['user_type']}: Avg {r['avg_score']} | NPS {nps} | Detractors {self._fmt_pct(r['detractor_pct'])} | n={r['cnt']}")
        return "\n".join(lines)

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    def describe_nps(self) -> str:
        """Run full Layer 1 descriptive analysis of NPS scores."""

        print("Running NPS Descriptive Analysis (Layer 1)...")
        print("Gathering data across all dimensions...")

        # Gather all data
        sections = [
            self._get_overall_summary(),
            self._get_quarterly_trend(),
            self._get_by_transaction_role(),
            self._get_by_bidding_intensity(),
            self._get_by_win_rate(),
            self._get_by_bid_value_tier(),
            self._get_by_bid_recency(),
            self._get_by_engagement_level(),
            self._get_by_account_tenure(),
            self._get_by_region(),
            self._get_copart_comparison(),
        ]

        nps_data = "\n\n".join(sections)

        framework = """
ANALYSIS FRAMEWORK:

You are analyzing NPS (Net Promoter Score) data for Purple Wave, a no-reserve auction company.
NPS = % Promoters (score 9-10) minus % Detractors (score 0-6). Passives score 7-8.

YOUR TASK: Identify the 5-7 most important patterns in this data.

PRIORITIZE INSIGHTS BY:
1. Largest gaps between segments (which groups are happiest vs least happy?)
2. Trends over time (is NPS improving or declining?)
3. Bidding behavior patterns (does bidding activity predict satisfaction?)
4. Actionable findings (what could leadership do about this?)

FOR EACH INSIGHT:
- Lead with the finding, not the dimension
- Include specific numbers (percentages, scores)
- Compare segments where relevant
- Note sample sizes if a finding is based on small numbers

TONE: Direct, data-driven, written for leadership. No hedging.
"""

        print("Generating insights...")
        result = self.describe(nps_data=nps_data, framework=framework)
        return result.analysis