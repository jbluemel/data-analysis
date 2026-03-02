import dspy
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from shared.database import AuctionDatabase
from typing import Dict, List, Tuple


class DiagnoseNPSDrivers(dspy.Signature):
    """Synthesize statistical findings about what drives NPS detraction at Purple Wave."""

    statistical_findings = dspy.InputField(desc="Results from statistical tests including significance tests, logistic regression, and effect sizes")
    framework = dspy.InputField(desc="Instructions for how to synthesize and present findings")
    diagnosis = dspy.OutputField(desc="Clear diagnosis of what drives detraction, what's statistically significant vs noise, and what the data says about root causes")


DIAGNOSTIC_FRAMEWORK = """You are analyzing statistical test results from Purple Wave's NPS data to determine
WHY certain customer segments have higher detractor rates. Your job is to separate signal from noise.

Rules:
- Only cite findings where p < 0.05. If p > 0.05, explicitly say the difference is NOT statistically significant.
- Report effect sizes, not just p-values. A tiny difference can be statistically significant with large samples.
- For logistic regression coefficients, explain in plain English what each driver means.
- Rank findings by practical impact (effect size × affected population), not just statistical significance.
- Be direct about what the data does and doesn't support.
- Identify the 3-5 most actionable root causes.
- Note any surprising non-findings (things you'd expect to matter but don't).

Tone: Direct, data-driven, for a VP of IT presenting to leadership.
Format: Numbered findings, each with the statistical evidence and business interpretation.
"""


class NPSDiagnostic(dspy.Module):
    """Layer 2: Statistical diagnosis of what drives NPS detraction."""

    def __init__(self, db: AuctionDatabase):
        super().__init__()
        self.db = db
        self.synthesize = dspy.ChainOfThought(DiagnoseNPSDrivers)

    def query(self, sql: str) -> List[Dict]:
        return self.db.query(sql)

    def _load_analysis_data(self) -> pd.DataFrame:
        """Load full NPS dataset for statistical analysis."""
        rows = self.query("""
            SELECT entity_id, nps_score, nps_score_label,
                nps_submitted_fiscal_year, nps_submitted_fiscal_quarter,
                transaction_role, bidding_intensity_tier, win_rate_tier,
                avg_bid_value_tier, bid_recency, account_tenure_tier,
                engagement_level, region_name, copart_user,
                bought_item_count, sold_item_count,
                bought_hammer_total, sold_hammer_total,
                lifetime_bids, wins, win_rate_pct, avg_highest_bid,
                serious_bid_pct, account_tenure_days
            FROM nps_enriched
            WHERE nps_score_label IN ('detractor', 'promoter')
        """)
        df = pd.DataFrame(rows)
        df['is_detractor'] = (df['nps_score_label'] == 'detractor').astype(int)
        return df

    # =========================================================================
    # TEST 1: Quarter-over-quarter detractor rate changes
    # =========================================================================

    def _test_quarterly_spike(self, df: pd.DataFrame) -> str:
        """Test if FY26 Q2 detractor spike is statistically significant."""
        print("  Test 1: Quarterly detractor rate significance...")

        results = []

        # Group by fiscal year + quarter
        quarterly = df.groupby(['nps_submitted_fiscal_year', 'nps_submitted_fiscal_quarter']).agg(
            total=('is_detractor', 'count'),
            detractors=('is_detractor', 'sum')
        ).reset_index()
        quarterly['rate'] = quarterly['detractors'] / quarterly['total']

        results.append("QUARTERLY DETRACTOR RATES:")
        for _, row in quarterly.iterrows():
            results.append(f"  FY{int(row['nps_submitted_fiscal_year'])} Q{int(row['nps_submitted_fiscal_quarter'])}: "
                         f"{row['rate']:.1%} ({int(row['detractors'])}/{int(row['total'])})")

        # Compare FY26 Q2 vs all other quarters combined
        fy26q2 = df[(df['nps_submitted_fiscal_year'] == 2026) & (df['nps_submitted_fiscal_quarter'] == 2)]
        others = df[~((df['nps_submitted_fiscal_year'] == 2026) & (df['nps_submitted_fiscal_quarter'] == 2))]

        # Exclude Q3 FY26 (incomplete)
        others = others[~((others['nps_submitted_fiscal_year'] == 2026) & (others['nps_submitted_fiscal_quarter'] == 3))]

        fy26q2_det = fy26q2['is_detractor'].sum()
        fy26q2_total = len(fy26q2)
        others_det = others['is_detractor'].sum()
        others_total = len(others)

        # Two-proportion z-test
        p1 = fy26q2_det / fy26q2_total
        p2 = others_det / others_total
        p_pool = (fy26q2_det + others_det) / (fy26q2_total + others_total)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/fy26q2_total + 1/others_total))
        z_stat = (p1 - p2) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        results.append(f"\nFY26 Q2 vs ALL OTHER QUARTERS:")
        results.append(f"  FY26 Q2 detractor rate: {p1:.1%} ({fy26q2_det}/{fy26q2_total})")
        results.append(f"  Other quarters rate: {p2:.1%} ({others_det}/{others_total})")
        results.append(f"  Difference: {(p1-p2):.1%} percentage points")
        results.append(f"  Z-statistic: {z_stat:.3f}, p-value: {p_value:.4f}")
        results.append(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")

        # Also compare FY26 Q2 vs FY25 Q2 (same quarter prior year)
        fy25q2 = df[(df['nps_submitted_fiscal_year'] == 2025) & (df['nps_submitted_fiscal_quarter'] == 2)]
        if len(fy25q2) > 0:
            fy25q2_det = fy25q2['is_detractor'].sum()
            fy25q2_total = len(fy25q2)
            p3 = fy25q2_det / fy25q2_total
            p_pool2 = (fy26q2_det + fy25q2_det) / (fy26q2_total + fy25q2_total)
            se2 = np.sqrt(p_pool2 * (1 - p_pool2) * (1/fy26q2_total + 1/fy25q2_total))
            z_stat2 = (p1 - p3) / se2 if se2 > 0 else 0
            p_value2 = 2 * (1 - stats.norm.cdf(abs(z_stat2)))

            results.append(f"\nFY26 Q2 vs FY25 Q2 (year-over-year):")
            results.append(f"  FY25 Q2 rate: {p3:.1%} ({fy25q2_det}/{fy25q2_total})")
            results.append(f"  FY26 Q2 rate: {p1:.1%} ({fy26q2_det}/{fy26q2_total})")
            results.append(f"  Difference: {(p1-p3):.1%} percentage points")
            results.append(f"  Z-statistic: {z_stat2:.3f}, p-value: {p_value2:.4f}")
            results.append(f"  {'SIGNIFICANT' if p_value2 < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")

        return "\n".join(results)

    # =========================================================================
    # TEST 2: Segment proportion tests
    # =========================================================================

    def _test_segment_differences(self, df: pd.DataFrame) -> str:
        """Test if detractor rates differ significantly across key segments."""
        print("  Test 2: Segment detractor rate comparisons...")

        results = ["SEGMENT DETRACTOR RATE COMPARISONS:"]

        comparisons = [
            {
                "name": "Power bidders vs casual bidders",
                "group_a": df[df['bidding_intensity_tier'] == 'power_bidder (1000+)'],
                "group_b": df[df['bidding_intensity_tier'] == 'casual (< 10 bids)'],
                "label_a": "Power bidders (1000+)",
                "label_b": "Casual (1-9 bids)"
            },
            {
                "name": "Buyer+seller vs buyer-only",
                "group_a": df[df['transaction_role'] == 'buyer_and_seller'],
                "group_b": df[df['transaction_role'] == 'buyer_only'],
                "label_a": "Buyer and seller",
                "label_b": "Buyer only"
            },
            {
                "name": "Veteran (5+ yr) vs new accounts",
                "group_a": df[df['account_tenure_tier'] == 'veteran (5+ years)'],
                "group_b": df[df['account_tenure_tier'].isin(['new (< 90 days)', 'early (3-12 months)'])],
                "label_a": "Veteran (5+ years)",
                "label_b": "New/early (< 2 years)"
            },
            {
                "name": "Lapsed bidders vs recent bidders",
                "group_a": df[df['bid_recency'] == 'bid_over_year_ago'],
                "group_b": df[df['bid_recency'] == 'bid_last_30_days'],
                "label_a": "Lapsed (bid 1+ year ago)",
                "label_b": "Active (bid last 30 days)"
            },
            {
                "name": "Copart users vs non-Copart",
                "group_a": df[df['copart_user'] == True],
                "group_b": df[df['copart_user'] == False],
                "label_a": "Copart user",
                "label_b": "Non-Copart user"
            },
            {
                "name": "Premium bidders ($50k+) vs micro bidders (< $500)",
                "group_a": df[df['avg_bid_value_tier'] == 'premium ($50k+)'],
                "group_b": df[df['avg_bid_value_tier'] == 'micro (< $1k)'],
                "label_a": "Premium ($50k+)",
                "label_b": "Micro (< $500)"
            },
        ]

        for comp in comparisons:
            a = comp['group_a']
            b = comp['group_b']

            if len(a) < 5 or len(b) < 5:
                results.append(f"\n--- {comp['name']} ---")
                results.append(f"  SKIPPED: insufficient sample size (a={len(a)}, b={len(b)})")
                continue

            a_det = a['is_detractor'].sum()
            b_det = b['is_detractor'].sum()
            a_total = len(a)
            b_total = len(b)
            p_a = a_det / a_total
            p_b = b_det / b_total

            # Two-proportion z-test
            p_pool = (a_det + b_det) / (a_total + b_total)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/a_total + 1/b_total))
            z_stat = (p_a - p_b) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            # Effect size (Cohen's h)
            h = 2 * (np.arcsin(np.sqrt(p_a)) - np.arcsin(np.sqrt(p_b)))

            results.append(f"\n--- {comp['name']} ---")
            results.append(f"  {comp['label_a']}: {p_a:.1%} detractor rate ({a_det}/{a_total})")
            results.append(f"  {comp['label_b']}: {p_b:.1%} detractor rate ({b_det}/{b_total})")
            results.append(f"  Difference: {(p_a-p_b):.1%} percentage points")
            results.append(f"  Z-statistic: {z_stat:.3f}, p-value: {p_value:.4f}")
            results.append(f"  Cohen's h effect size: {abs(h):.3f} ({'small' if abs(h) < 0.2 else 'medium' if abs(h) < 0.5 else 'large'})")
            results.append(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")

        return "\n".join(results)

    # =========================================================================
    # TEST 3: Continuous variable comparisons
    # =========================================================================

    def _test_continuous_drivers(self, df: pd.DataFrame) -> str:
        """Compare continuous metrics between detractors and promoters."""
        print("  Test 3: Continuous variable comparisons (detractors vs promoters)...")

        results = ["CONTINUOUS VARIABLE COMPARISONS (Detractors vs Promoters):"]

        detractors = df[df['is_detractor'] == 1]
        promoters = df[df['is_detractor'] == 0]

        metrics = [
            ('lifetime_bids', 'Lifetime bids'),
            ('wins', 'Total wins'),
            ('win_rate_pct', 'Win rate %'),
            ('avg_highest_bid', 'Avg highest bid $'),
            ('serious_bid_pct', 'Serious bid %'),
            ('bought_item_count', 'Items bought'),
            ('sold_item_count', 'Items sold'),
            ('bought_hammer_total', 'Total $ bought'),
            ('sold_hammer_total', 'Total $ sold'),
            ('account_tenure_days', 'Account tenure (days)'),
        ]

        for col, label in metrics:
            d_vals = pd.to_numeric(detractors[col], errors='coerce').dropna()
            p_vals = pd.to_numeric(promoters[col], errors='coerce').dropna()

            if len(d_vals) < 5 or len(p_vals) < 5:
                continue

            d_mean = d_vals.mean()
            p_mean = p_vals.mean()
            d_median = d_vals.median()
            p_median = p_vals.median()

            # Mann-Whitney U test (non-parametric, handles skewed distributions)
            u_stat, p_value = stats.mannwhitneyu(d_vals, p_vals, alternative='two-sided')

            # Effect size (rank-biserial correlation)
            n1, n2 = len(d_vals), len(p_vals)
            r = 1 - (2 * u_stat) / (n1 * n2)

            results.append(f"\n--- {label} ---")
            results.append(f"  Detractors: mean={d_mean:,.1f}, median={d_median:,.1f} (n={n1})")
            results.append(f"  Promoters:  mean={p_mean:,.1f}, median={p_median:,.1f} (n={n2})")
            results.append(f"  Mann-Whitney U p-value: {p_value:.4f}")
            results.append(f"  Effect size (rank-biserial r): {abs(r):.3f} ({'small' if abs(r) < 0.1 else 'medium' if abs(r) < 0.3 else 'large'})")
            results.append(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")
            if d_mean > p_mean:
                results.append(f"  Direction: Detractors have HIGHER {label}")
            else:
                results.append(f"  Direction: Detractors have LOWER {label}")

        return "\n".join(results)

    # =========================================================================
    # TEST 4: Logistic regression - multivariate drivers
    # =========================================================================

    def _test_logistic_regression(self, df: pd.DataFrame) -> str:
        """Logistic regression to identify which factors predict detractor status."""
        print("  Test 4: Logistic regression (multivariate driver analysis)...")

        results = ["LOGISTIC REGRESSION - MULTIVARIATE DETRACTOR DRIVERS:"]

        # Prepare features
        feature_cols = [
            'transaction_role', 'bidding_intensity_tier', 'win_rate_tier',
            'bid_recency', 'account_tenure_tier', 'engagement_level',
        ]

        numeric_cols = [
            'win_rate_pct', 'avg_highest_bid', 'lifetime_bids',
            'bought_item_count', 'sold_item_count', 'account_tenure_days'
        ]

        model_df = df.copy()

        # Encode categoricals
        encoders = {}
        encoded_features = []
        for col in feature_cols:
            model_df[col] = model_df[col].fillna('unknown')
            le = LabelEncoder()
            model_df[f'{col}_enc'] = le.fit_transform(model_df[col])
            encoders[col] = le
            encoded_features.append(f'{col}_enc')

        # Clean numerics
        for col in numeric_cols:
            model_df[col] = pd.to_numeric(model_df[col], errors='coerce').fillna(0)

        all_features = encoded_features + numeric_cols
        X = model_df[all_features].values
        y = model_df['is_detractor'].values

        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit model
        model = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=1.0)
        model.fit(X_scaled, y)

        # Score
        accuracy = model.score(X_scaled, y)
        results.append(f"\nModel accuracy: {accuracy:.1%}")
        results.append(f"Baseline (always predict promoter): {1 - y.mean():.1%}")
        results.append(f"Model lift over baseline: {accuracy - (1 - y.mean()):.1%}")

        # Feature importance (coefficients)
        coef_df = pd.DataFrame({
            'feature': all_features,
            'coefficient': model.coef_[0],
            'abs_coefficient': np.abs(model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)

        results.append(f"\nFEATURE IMPORTANCE (by absolute coefficient):")
        results.append(f"  Positive coefficient = increases detractor probability")
        results.append(f"  Negative coefficient = decreases detractor probability")
        results.append("")

        for _, row in coef_df.iterrows():
            feature_name = row['feature'].replace('_enc', '')
            direction = "↑ detractor" if row['coefficient'] > 0 else "↓ detractor"
            results.append(f"  {feature_name:30s} coef={row['coefficient']:+.4f}  ({direction})")

        # Odds ratios for top features
        results.append(f"\nODDS RATIOS (top 5 drivers):")
        for _, row in coef_df.head(5).iterrows():
            feature_name = row['feature'].replace('_enc', '')
            odds_ratio = np.exp(row['coefficient'])
            results.append(f"  {feature_name}: OR={odds_ratio:.3f} "
                         f"({'increases' if odds_ratio > 1 else 'decreases'} odds by {abs(odds_ratio-1)*100:.1f}%)")

        return "\n".join(results)

    # =========================================================================
    # TEST 5: Interaction effects
    # =========================================================================

    def _test_interactions(self, df: pd.DataFrame) -> str:
        """Test specific interaction effects hypothesized from Layer 1."""
        print("  Test 5: Interaction effects...")

        results = ["INTERACTION EFFECTS:"]

        # Interaction 1: Power bidder + low win rate
        power_low_win = df[(df['bidding_intensity_tier'] == 'power_bidder (1000+)') &
                           (pd.to_numeric(df['win_rate_pct'], errors='coerce') < 5)]
        power_high_win = df[(df['bidding_intensity_tier'] == 'power_bidder (1000+)') &
                            (pd.to_numeric(df['win_rate_pct'], errors='coerce') >= 10)]

        if len(power_low_win) >= 5 and len(power_high_win) >= 5:
            a_rate = power_low_win['is_detractor'].mean()
            b_rate = power_high_win['is_detractor'].mean()
            # Fisher exact test for small samples
            table = np.array([
                [power_low_win['is_detractor'].sum(), len(power_low_win) - power_low_win['is_detractor'].sum()],
                [power_high_win['is_detractor'].sum(), len(power_high_win) - power_high_win['is_detractor'].sum()]
            ])
            _, p_val = stats.fisher_exact(table)

            results.append(f"\n--- Power bidder + low win rate (<5%) vs high win rate (>=10%) ---")
            results.append(f"  Low win rate: {a_rate:.1%} detractor rate (n={len(power_low_win)})")
            results.append(f"  High win rate: {b_rate:.1%} detractor rate (n={len(power_high_win)})")
            results.append(f"  Fisher exact p-value: {p_val:.4f}")
            results.append(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")

        # Interaction 2: Buyer+seller + veteran
        dual_veteran = df[(df['transaction_role'] == 'buyer_and_seller') &
                          (df['account_tenure_tier'] == 'veteran (5+ years)')]
        buyer_veteran = df[(df['transaction_role'] == 'buyer_only') &
                           (df['account_tenure_tier'] == 'veteran (5+ years)')]

        if len(dual_veteran) >= 5 and len(buyer_veteran) >= 5:
            a_rate = dual_veteran['is_detractor'].mean()
            b_rate = buyer_veteran['is_detractor'].mean()
            table = np.array([
                [dual_veteran['is_detractor'].sum(), len(dual_veteran) - dual_veteran['is_detractor'].sum()],
                [buyer_veteran['is_detractor'].sum(), len(buyer_veteran) - buyer_veteran['is_detractor'].sum()]
            ])
            _, p_val = stats.fisher_exact(table)

            results.append(f"\n--- Veteran buyer+seller vs veteran buyer-only ---")
            results.append(f"  Veteran buyer+seller: {a_rate:.1%} detractor rate (n={len(dual_veteran)})")
            results.append(f"  Veteran buyer-only: {b_rate:.1%} detractor rate (n={len(buyer_veteran)})")
            results.append(f"  Fisher exact p-value: {p_val:.4f}")
            results.append(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")

        # Interaction 3: Lapsed + high prior engagement
        lapsed_high = df[(df['bid_recency'] == 'bid_over_year_ago') &
                         (df['engagement_level'].isin(['high', 'very_high']))]
        lapsed_low = df[(df['bid_recency'] == 'bid_over_year_ago') &
                        (df['engagement_level'].isin(['low', 'medium']))]

        if len(lapsed_high) >= 5 and len(lapsed_low) >= 5:
            a_rate = lapsed_high['is_detractor'].mean()
            b_rate = lapsed_low['is_detractor'].mean()
            table = np.array([
                [lapsed_high['is_detractor'].sum(), len(lapsed_high) - lapsed_high['is_detractor'].sum()],
                [lapsed_low['is_detractor'].sum(), len(lapsed_low) - lapsed_low['is_detractor'].sum()]
            ])
            _, p_val = stats.fisher_exact(table)

            results.append(f"\n--- Lapsed + high engagement vs lapsed + low engagement ---")
            results.append(f"  Lapsed high-engagement: {a_rate:.1%} detractor rate (n={len(lapsed_high)})")
            results.append(f"  Lapsed low-engagement: {b_rate:.1%} detractor rate (n={len(lapsed_low)})")
            results.append(f"  Fisher exact p-value: {p_val:.4f}")
            results.append(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")

        # Interaction 4: Recent buyer who also sells vs recent buyer only
        recent_dual = df[(df['bid_recency'] == 'bid_last_30_days') &
                         (df['transaction_role'] == 'buyer_and_seller')]
        recent_buyer = df[(df['bid_recency'] == 'bid_last_30_days') &
                          (df['transaction_role'] == 'buyer_only')]

        if len(recent_dual) >= 5 and len(recent_buyer) >= 5:
            a_rate = recent_dual['is_detractor'].mean()
            b_rate = recent_buyer['is_detractor'].mean()
            table = np.array([
                [recent_dual['is_detractor'].sum(), len(recent_dual) - recent_dual['is_detractor'].sum()],
                [recent_buyer['is_detractor'].sum(), len(recent_buyer) - recent_buyer['is_detractor'].sum()]
            ])
            _, p_val = stats.fisher_exact(table)

            results.append(f"\n--- Active buyer+seller vs active buyer-only ---")
            results.append(f"  Active buyer+seller: {a_rate:.1%} detractor rate (n={len(recent_dual)})")
            results.append(f"  Active buyer-only: {b_rate:.1%} detractor rate (n={len(recent_buyer)})")
            results.append(f"  Fisher exact p-value: {p_val:.4f}")
            results.append(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")

        return "\n".join(results)

    # =========================================================================
    # TEST 6: Transaction proximity analysis
    # =========================================================================

    def _test_transaction_proximity(self) -> str:
        """Test if having a recent transaction before NPS correlates with detractor status."""
        print("  Test 6: Transaction proximity to NPS score...")

        results = ["TRANSACTION PROXIMITY ANALYSIS:"]

        # Get all detractors/promoters with their nearest transaction
        rows = self.query("""
            SELECT n.entity_id, n.nps_score_label,
                t.txn_role, t.days_from_nps, t.contract_price, t.txn_proximity_label
            FROM nps_enriched n
            LEFT JOIN nps_nearest_transactions t
                ON n.entity_id = t.entity_id AND t.txn_rank = 1 AND t.days_from_nps >= 0
            WHERE n.nps_score_label IN ('detractor', 'promoter')
        """)

        df = pd.DataFrame(rows)
        df['is_detractor'] = (df['nps_score_label'] == 'detractor').astype(int)
        df['has_recent_txn'] = df['days_from_nps'].notna().astype(int)

        # Compare detractor rate: has recent transaction vs no transaction
        with_txn = df[df['has_recent_txn'] == 1]
        without_txn = df[df['has_recent_txn'] == 0]

        if len(with_txn) >= 5 and len(without_txn) >= 5:
            a_rate = with_txn['is_detractor'].mean()
            b_rate = without_txn['is_detractor'].mean()

            p_pool = (with_txn['is_detractor'].sum() + without_txn['is_detractor'].sum()) / (len(with_txn) + len(without_txn))
            se = np.sqrt(p_pool * (1 - p_pool) * (1/len(with_txn) + 1/len(without_txn)))
            z_stat = (a_rate - b_rate) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            results.append(f"\n--- Has transaction before NPS vs no transaction in data ---")
            results.append(f"  With transaction: {a_rate:.1%} detractor rate (n={len(with_txn)})")
            results.append(f"  Without transaction: {b_rate:.1%} detractor rate (n={len(without_txn)})")
            results.append(f"  Z-statistic: {z_stat:.3f}, p-value: {p_value:.4f}")
            results.append(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")

        # Among those with transactions, does proximity matter?
        txn_df = df[df['has_recent_txn'] == 1].copy()
        txn_df['days_from_nps'] = pd.to_numeric(txn_df['days_from_nps'], errors='coerce')

        if len(txn_df) >= 20:
            within_30 = txn_df[txn_df['days_from_nps'] <= 30]
            over_30 = txn_df[txn_df['days_from_nps'] > 30]

            if len(within_30) >= 5 and len(over_30) >= 5:
                a_rate = within_30['is_detractor'].mean()
                b_rate = over_30['is_detractor'].mean()

                table = np.array([
                    [within_30['is_detractor'].sum(), len(within_30) - within_30['is_detractor'].sum()],
                    [over_30['is_detractor'].sum(), len(over_30) - over_30['is_detractor'].sum()]
                ])
                _, p_val = stats.fisher_exact(table)

                results.append(f"\n--- Transaction within 30 days vs over 30 days before NPS ---")
                results.append(f"  Within 30 days: {a_rate:.1%} detractor rate (n={len(within_30)})")
                results.append(f"  Over 30 days: {b_rate:.1%} detractor rate (n={len(over_30)})")
                results.append(f"  Fisher exact p-value: {p_val:.4f}")
                results.append(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")

        # Buyer vs seller transactions among detractors
        det_txns = txn_df[txn_df['is_detractor'] == 1]
        if len(det_txns) >= 5:
            buyer_txns = det_txns[det_txns['txn_role'] == 'buyer']
            seller_txns = det_txns[det_txns['txn_role'] == 'seller']
            results.append(f"\n--- Transaction role among detractors with transactions ---")
            results.append(f"  Buyer transactions: {len(buyer_txns)}")
            results.append(f"  Seller transactions: {len(seller_txns)}")
            if len(buyer_txns) > 0:
                results.append(f"  Buyer txn median price: ${pd.to_numeric(buyer_txns['contract_price'], errors='coerce').median():,.0f}")
            if len(seller_txns) > 0:
                results.append(f"  Seller txn median price: ${pd.to_numeric(seller_txns['contract_price'], errors='coerce').median():,.0f}")

        return "\n".join(results)

    # =========================================================================
    # MAIN DIAGNOSTIC METHOD
    # =========================================================================

    def diagnose(self) -> str:
        """Run all diagnostic tests and synthesize findings."""
        print("Running NPS Diagnostic Analysis (Layer 2)...")
        print("Loading data...")

        df = self._load_analysis_data()
        print(f"Loaded {len(df)} records ({df['is_detractor'].sum()} detractors, {(1-df['is_detractor']).sum():.0f} promoters)\n")

        # Run all tests
        all_findings = []

        test1 = self._test_quarterly_spike(df)
        all_findings.append(test1)
        print()

        test2 = self._test_segment_differences(df)
        all_findings.append(test2)
        print()

        test3 = self._test_continuous_drivers(df)
        all_findings.append(test3)
        print()

        test4 = self._test_logistic_regression(df)
        all_findings.append(test4)
        print()

        test5 = self._test_interactions(df)
        all_findings.append(test5)
        print()

        test6 = self._test_transaction_proximity()
        all_findings.append(test6)
        print()

        combined_findings = "\n\n" + "=" * 60 + "\n\n".join(all_findings)

        # Save raw findings
        os.makedirs("reports/nps", exist_ok=True)
        with open("reports/nps/nps_diagnostic_raw.txt", "w") as f:
            f.write(combined_findings)
        print("Raw findings saved: reports/nps/nps_diagnostic_raw.txt")

        # LLM synthesis
        print("\nSynthesizing findings with LLM...")
        result = self.synthesize(
            statistical_findings=combined_findings,
            framework=DIAGNOSTIC_FRAMEWORK
        )

        diagnosis = result.diagnosis

        with open("reports/nps/nps_diagnostic_report.txt", "w") as f:
            f.write("NPS DIAGNOSTIC ANALYSIS (LAYER 2)\n")
            f.write("=" * 60 + "\n\n")
            f.write(diagnosis)
        print("Diagnostic report saved: reports/nps/nps_diagnostic_report.txt")

        print("\n" + "=" * 60)
        print("DIAGNOSIS:")
        print("=" * 60)
        print(diagnosis)

        return diagnosis


import os