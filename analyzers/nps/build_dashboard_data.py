"""
NPS Dashboard Data Builder
===========================
Queries nps_enriched via shared.database.AuctionDatabase,
runs descriptive + diagnostic + predictive + prescriptive analysis,
and injects structured JSON directly into the HTML report template.

Usage:
    From run_analysis.py: --dashboard flag
    Standalone: python -m analyzers.nps.build_dashboard_data
"""

import json
import argparse
import sys
import os
from datetime import datetime, date
from decimal import Decimal

import dspy
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from shared.database import AuctionDatabase


# Paths relative to project root (data-analysis/)
HTML_TEMPLATE = "reports/nps/nps_report.html"
TABLE = "nps_enriched"
NPS_GOAL = 65
REPORT_FY = 2026

# Markers in the HTML template
DATA_START = "/*__NPS_DATA__*/"
DATA_END = "/*__END_DATA__*/"


# ============================================================
# DSPY SIGNATURES
# ============================================================

class SummarizeComments(dspy.Signature):
    """Summarize NPS survey comments from a specific customer category (promoter, passive, or detractor).
    Identify the top 3-5 themes and write a concise 2-3 sentence narrative summary.
    Be specific about what customers mention — name concrete topics, not vague generalities."""

    category: str = dspy.InputField(desc="NPS category: promoter, passive, or detractor")
    comment_count: int = dspy.InputField(desc="Total number of comments in this category")
    comments: str = dspy.InputField(desc="Sample comments from this category, one per line")
    summary: str = dspy.OutputField(desc="2-3 sentence narrative summary of key themes")


class DiagnosticNarrative(dspy.Signature):
    """Analyze NPS diagnostic data powered by XGBoost + SHAP analysis.
    Write a 1-2 sentence business-friendly narrative for each SHAP feature explaining what it means for the company.
    Then write an overall synthesis connecting the patterns.
    Avoid technical jargon like 'SHAP values'. Instead say things like 'our analysis shows' or 'the data indicates'.
    Be specific with numbers. Focus on what the business should do about it."""

    report_fy: int = dspy.InputField(desc="Fiscal year being reported on")
    nps_goal: int = dspy.InputField(desc="Company NPS goal")
    segment_data: str = dspy.InputField(desc="SHAP importance rankings, value details, and segment breakdowns")
    engagement_paradox: str = dspy.InputField(desc="Engagement level vs detractor rate data")
    bidding_comparison: str = dspy.InputField(desc="Bidding behavior context")
    feature_list: str = dspy.InputField(desc="Comma-separated list of top 5 SHAP feature names. Write a narrative for each one.")
    narratives: str = dspy.OutputField(desc="JSON with 'overall_synthesis' (3-4 sentences, business-friendly) and 'feature_narratives' (object with key for EACH feature, 1-2 sentence business-friendly narrative each)")


def clean_label(raw: str) -> str:
    """Convert field names like 'very_high' to 'Very High'."""
    if not raw:
        return "Unknown"
    return raw.replace("_", " ").title()


# ============================================================
# HELPERS
# ============================================================

class JSONEncoder(json.JSONEncoder):
    """Handle numpy/pandas/decimal types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 4)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)


def safe_pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def nps_from_counts(promoters, detractors, total):
    if total == 0:
        return 0.0
    return round(100.0 * (promoters - detractors) / total, 1)


def chi_square_test(df, group_col, label_col="nps_score_label"):
    """Run chi-square test of independence between group and NPS label."""
    ct = pd.crosstab(df[group_col], df[label_col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": None, "p_value": None, "significant": False}
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    return {
        "chi2": round(float(chi2), 2),
        "p_value": round(float(p), 6),
        "dof": int(dof),
        "significant": p < 0.05,
    }


# ============================================================
# MAIN BUILDER CLASS
# ============================================================

class NPSDashboardBuilder:
    """Builds structured JSON for the NPS HTML dashboard."""

    def __init__(self, db: AuctionDatabase):
        self.db = db

    def _query_df(self, sql: str) -> pd.DataFrame:
        """Execute SQL via AuctionDatabase and return DataFrame."""
        rows = self.db.query(sql)
        return pd.DataFrame(rows)

    # =========================================================================
    # LAYER 1: DESCRIPTIVE
    # =========================================================================

    def _build_descriptive(self) -> dict:
        """Headline numbers, distributions, trends — FY26 focused."""
        print("  [1/4] Building descriptive data...")
        result = {}

        # ----- Fiscal year summary (both years for comparison) -----
        fy = self._query_df(f"""
            SELECT
                nps_submitted_fiscal_year as fy,
                COUNT(*) as responses,
                COUNT(DISTINCT entity_id) as unique_accounts,
                ROUND(AVG(nps_score)::numeric, 1) as avg_score,
                SUM(CASE WHEN nps_score_label = 'promoter' THEN 1 ELSE 0 END) as promoters,
                SUM(CASE WHEN nps_score_label = 'passive' THEN 1 ELSE 0 END) as passives,
                SUM(CASE WHEN nps_score_label = 'detractor' THEN 1 ELSE 0 END) as detractors,
                SUM(CASE WHEN has_comment THEN 1 ELSE 0 END) as with_comments
            FROM {TABLE}
            GROUP BY nps_submitted_fiscal_year
            ORDER BY nps_submitted_fiscal_year
        """)
        fy_data = []
        for _, row in fy.iterrows():
            total = row["responses"]
            fy_data.append({
                "fiscal_year": int(row["fy"]),
                "responses": int(total),
                "unique_accounts": int(row["unique_accounts"]),
                "avg_score": float(row["avg_score"]),
                "promoters": int(row["promoters"]),
                "passives": int(row["passives"]),
                "detractors": int(row["detractors"]),
                "promoter_pct": safe_pct(row["promoters"], total),
                "detractor_pct": safe_pct(row["detractors"], total),
                "nps": nps_from_counts(row["promoters"], row["detractors"], total),
                "comment_pct": safe_pct(row["with_comments"], total),
            })
        result["by_fiscal_year"] = fy_data
        result["nps_goal"] = NPS_GOAL
        result["report_fy"] = REPORT_FY

        # ----- FY26 headline (current year) -----
        current = next((f for f in fy_data if f["fiscal_year"] == REPORT_FY), fy_data[-1])
        prior = next((f for f in fy_data if f["fiscal_year"] == REPORT_FY - 1), None)
        result["headline"] = {
            "nps": current["nps"],
            "goal": NPS_GOAL,
            "goal_gap": round(current["nps"] - NPS_GOAL, 1),
            "responses": current["responses"],
            "unique_accounts": current["unique_accounts"],
            "promoter_pct": current["promoter_pct"],
            "passive_pct": safe_pct(current["passives"], current["responses"]),
            "detractor_pct": current["detractor_pct"],
            "avg_score": current["avg_score"],
            "comment_pct": current["comment_pct"],
            "nps_change": round(current["nps"] - prior["nps"], 1) if prior else None,
            "detractor_change": round(current["detractor_pct"] - prior["detractor_pct"], 1) if prior else None,
        }

        # ----- Score distribution (FY26 only) -----
        dist = self._query_df(f"""
            SELECT nps_score, COUNT(*) as cnt
            FROM {TABLE}
            WHERE nps_submitted_fiscal_year = {REPORT_FY}
            GROUP BY nps_score ORDER BY nps_score
        """)
        result["score_distribution"] = [
            {"score": int(r["nps_score"]), "count": int(r["cnt"])}
            for _, r in dist.iterrows()
        ]

        # ----- Quarterly trend (all quarters, both years for context) -----
        qt = self._query_df(f"""
            SELECT
                nps_submitted_fiscal_year as fy,
                nps_submitted_fiscal_quarter as fq,
                COUNT(*) as responses,
                SUM(CASE WHEN nps_score_label = 'promoter' THEN 1 ELSE 0 END) as promoters,
                SUM(CASE WHEN nps_score_label = 'detractor' THEN 1 ELSE 0 END) as detractors
            FROM {TABLE}
            GROUP BY 1, 2 ORDER BY 1, 2
        """)
        result["quarterly_trend"] = [
            {
                "period": f"FY{int(r['fy'])} Q{int(r['fq'])}",
                "fiscal_year": int(r["fy"]),
                "responses": int(r["responses"]),
                "nps": nps_from_counts(r["promoters"], r["detractors"], r["responses"]),
                "detractor_pct": safe_pct(r["detractors"], r["responses"]),
                "promoter_pct": safe_pct(r["promoters"], r["responses"]),
            }
            for _, r in qt.iterrows()
        ]

        # ----- Source breakdown (FY26 only) -----
        src = self._query_df(f"""
            SELECT
                nps_source,
                COUNT(*) as cnt,
                ROUND(AVG(nps_score)::numeric, 1) as avg_score,
                SUM(CASE WHEN nps_score_label = 'detractor' THEN 1 ELSE 0 END) as detractors
            FROM {TABLE}
            WHERE nps_submitted_fiscal_year = {REPORT_FY}
            GROUP BY 1 ORDER BY 1
        """)
        result["by_source"] = [
            {
                "source": r["nps_source"],
                "count": int(r["cnt"]),
                "avg_score": float(r["avg_score"]),
                "detractor_pct": safe_pct(r["detractors"], r["cnt"]),
            }
            for _, r in src.iterrows()
        ]

        # ----- Comment summaries per category (FY26, DSPy) -----
        result["comment_categories"] = self._build_comment_summaries()

        return result

    def _build_comment_summaries(self) -> list:
        """Build stats and DSPy narrative for each NPS category's comments."""
        print("    → Summarizing comments via LLM...")
        summarizer = dspy.ChainOfThought(SummarizeComments)
        categories = []

        for label in ["promoter", "passive", "detractor"]:
            # Stats for this category
            stats_row = self._query_df(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN has_comment THEN 1 ELSE 0 END) as with_comments,
                    ROUND(AVG(nps_score)::numeric, 1) as avg_score,
                    ROUND(AVG(COALESCE(bought_hammer_total, 0) + COALESCE(sold_hammer_total, 0))::numeric, 0) as avg_total_dollars
                FROM {TABLE}
                WHERE nps_submitted_fiscal_year = {REPORT_FY}
                  AND nps_score_label = '{label}'
            """)
            row = stats_row.iloc[0]
            total = int(row["total"])
            with_comments = int(row["with_comments"])

            # Get comments (prioritized by customer value)
            comments_df = self._query_df(f"""
                SELECT nps_comment
                FROM {TABLE}
                WHERE nps_submitted_fiscal_year = {REPORT_FY}
                  AND nps_score_label = '{label}'
                  AND has_comment = true
                ORDER BY (COALESCE(bought_hammer_total, 0) + COALESCE(sold_hammer_total, 0)) DESC
                LIMIT 50
            """)
            comment_texts = comments_df["nps_comment"].tolist()

            # DSPy summary
            summary_text = ""
            if comment_texts:
                try:
                    result = summarizer(
                        category=label,
                        comment_count=with_comments,
                        comments="\n".join(str(c) for c in comment_texts[:50]),
                    )
                    summary_text = result.summary
                    print(f"      ✓ {label}: {with_comments} comments summarized")
                except Exception as e:
                    summary_text = f"Summary unavailable: {e}"
                    print(f"      ✗ {label}: summary failed — {e}")
            else:
                summary_text = "No comments available for this category."

            categories.append({
                "label": label,
                "total_responses": total,
                "with_comments": with_comments,
                "comment_rate": safe_pct(with_comments, total),
                "avg_score": float(row["avg_score"]),
                "avg_total_dollars": float(row["avg_total_dollars"]) if row["avg_total_dollars"] else 0,
                "summary": summary_text,
            })

        return categories

    # =========================================================================
    # LAYER 2: DIAGNOSTIC
    # =========================================================================

    def _segment_analysis(self, df_display, df_stat, group_col, label):
        """Break down NPS by a grouping column (supporting detail).
        df_display: FY26 data for the numbers shown in the report.
        df_stat: full dataset for chi-square statistical testing.
        """
        segments = []
        for val, grp in df_display.groupby(group_col):
            total = len(grp)
            det = (grp["nps_score_label"] == "detractor").sum()
            pro = (grp["nps_score_label"] == "promoter").sum()
            segments.append({
                "segment": clean_label(str(val)) if val is not None else "Unknown",
                "count": int(total),
                "avg_score": round(float(grp["nps_score"].mean()), 1),
                "detractor_pct": float(safe_pct(det, total)),
                "promoter_pct": float(safe_pct(pro, total)),
                "nps": nps_from_counts(pro, det, total),
                "avg_bought_hammer": float(round(grp["bought_hammer_total"].fillna(0).mean(), 0)),
                "avg_sold_hammer": float(round(grp["sold_hammer_total"].fillna(0).mean(), 0)),
            })
        # Sort by NPS ascending (worst first)
        segments.sort(key=lambda x: x["nps"])
        # Chi-square on full dataset for statistical power
        stat_test = chi_square_test(df_stat.dropna(subset=[group_col]), group_col)
        return {"dimension": label, "group_column": group_col, "segments": segments, "statistical_test": stat_test}

    def _build_diagnostic(self) -> dict:
        """XGBoost + SHAP diagnostic: what drives detractor status?
        Model trained on full dataset for statistical power.
        Display numbers: FY26 only. Segment tables as supporting detail.
        """
        import xgboost as xgb
        import shap

        print("  [2/4] Building diagnostic data (XGBoost + SHAP)...")

        df_all = self._query_df(f"SELECT * FROM {TABLE}")
        df_fy = df_all[df_all["nps_submitted_fiscal_year"] == REPORT_FY].copy()
        result = {}
        result["report_fy"] = REPORT_FY

        # ---- XGBOOST + SHAP MODEL ----

        cat_features = [
            "transaction_role", "account_tenure_tier", "engagement_level",
            "customer_lifecycle_stage", "bidding_intensity_tier",
            "win_rate_tier", "avg_bid_value_tier", "bid_recency",
            "region_name",
            # nps_source excluded — survey channel is a methodology artifact,
            # not a business driver (iterable 29.5% vs posthog 9.6% detractor rate)
        ]
        num_features = [
            "bought_item_count", "sold_item_count",
            "bought_hammer_total", "sold_hammer_total",
            "items_bid_on", "lifetime_bids", "wins",
            "win_rate_pct", "avg_highest_bid", "serious_bids",
            "account_tenure_days", "total_transaction_count",
        ]
        cat_features = [c for c in cat_features if c in df_all.columns]
        num_features = [c for c in num_features if c in df_all.columns]

        # Prepare model dataframe (full dataset for training)
        model_df = df_all[cat_features + num_features + ["nps_score_label"]].copy()
        model_df["is_detractor"] = (model_df["nps_score_label"] == "detractor").astype(int)

        for col in cat_features:
            model_df[col] = model_df[col].fillna("unknown")
        for col in num_features:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0)

        # Label encode categoricals
        encoders = {}
        for col in cat_features:
            le = LabelEncoder()
            model_df[col + "_enc"] = le.fit_transform(model_df[col])
            encoders[col] = le

        feature_cols = [c + "_enc" for c in cat_features] + num_features
        feature_display_names = cat_features + num_features
        X = model_df[feature_cols].values
        y = model_df["is_detractor"].values

        # Train XGBoost
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        scale_pos = n_neg / n_pos if n_pos > 0 else 1

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_model.fit(X, y)

        # Cross-validation for model quality
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc = cross_val_score(xgb_model, X, y, cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(xgb_model, X, y, cv=cv, scoring="f1")

        result["model_quality"] = {
            "cv_accuracy": round(float(cv_acc.mean()), 4),
            "cv_accuracy_std": round(float(cv_acc.std()), 4),
            "cv_f1": round(float(cv_f1.mean()), 4),
            "cv_f1_std": round(float(cv_f1.std()), 4),
            "baseline_accuracy": round(float(1 - y.mean()), 4),
            "n_samples": int(len(y)),
            "n_detractors": int(n_pos),
        }
        print(f"    XGBoost CV accuracy: {cv_acc.mean():.1%} (baseline: {1 - y.mean():.1%}), F1: {cv_f1.mean():.1%}")

        # SHAP explanation
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X)

        # Global feature importance (mean |SHAP|)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_order = np.argsort(mean_abs_shap)[::-1]

        global_importance = []
        for idx in importance_order:
            fname = feature_display_names[idx]
            global_importance.append({
                "feature": clean_label(fname),
                "raw_feature": fname,
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
            })
        result["shap_importance"] = global_importance
        print(f"    Top SHAP features: {', '.join(g['feature'] for g in global_importance[:5])}")

        # Per-feature SHAP detail: top 5 features only
        top_features = importance_order[:5]
        shap_details = []
        for idx in top_features:
            fname = feature_display_names[idx]
            is_cat = fname in cat_features
            feature_shap = shap_values[:, idx]
            feature_vals = X[:, idx]

            detail = {
                "feature": clean_label(fname),
                "raw_feature": fname,
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
                "is_categorical": is_cat,
                "value_impacts": [],
            }

            if is_cat:
                le = encoders[fname]
                for encoded_val in sorted(np.unique(feature_vals)):
                    mask = feature_vals == encoded_val
                    if mask.sum() < 5:
                        continue
                    label_name = le.inverse_transform([int(encoded_val)])[0]
                    avg_shap = float(feature_shap[mask].mean())
                    detail["value_impacts"].append({
                        "value": clean_label(str(label_name)),
                        "count": int(mask.sum()),
                        "avg_shap": round(avg_shap, 4),
                        "direction": "toward detractor" if avg_shap > 0 else "away from detractor",
                    })
                detail["value_impacts"].sort(key=lambda x: x["avg_shap"], reverse=True)
            else:
                # Numeric: bucket into quartiles with actual range labels
                nonzero = feature_vals[feature_vals > 0]
                if len(nonzero) > 10:
                    quartiles = np.percentile(nonzero, [25, 50, 75])
                else:
                    quartiles = [0, 0, 0]

                # Format range values based on feature type
                def fmt_range_val(v, fname):
                    if v == float('-inf') or v == float('inf'):
                        return None
                    if any(kw in fname for kw in ['hammer', 'bid', 'sold', 'bought', 'price', 'value', 'dollar']):
                        if abs(v) >= 1000:
                            return f"${v/1000:,.0f}k"
                        return f"${v:,.0f}"
                    if 'days' in fname or 'tenure' in fname:
                        if v >= 365:
                            return f"{v/365:.1f}yr"
                        if v >= 30:
                            return f"{v/30:.0f}mo"
                        return f"{v:.0f}d"
                    if v >= 1000:
                        return f"{v/1000:,.1f}k"
                    return f"{v:,.0f}"

                for q_label, low, high in [
                    ("Low", float('-inf'), quartiles[0]),
                    ("Medium-Low", quartiles[0], quartiles[1]),
                    ("Medium-High", quartiles[1], quartiles[2]),
                    ("High", quartiles[2], float('inf')),
                ]:
                    mask = (feature_vals > low) & (feature_vals <= high)
                    if mask.sum() < 5:
                        continue
                    avg_shap = float(feature_shap[mask].mean())

                    # Build descriptive label with range
                    lo_str = fmt_range_val(low, fname)
                    hi_str = fmt_range_val(high, fname)
                    if lo_str and hi_str:
                        range_label = f"{q_label} ({lo_str}–{hi_str})"
                    elif hi_str:
                        range_label = f"{q_label} (≤{hi_str})"
                    elif lo_str:
                        range_label = f"{q_label} ({lo_str}+)"
                    else:
                        range_label = q_label

                    detail["value_impacts"].append({
                        "value": range_label,
                        "count": int(mask.sum()),
                        "avg_shap": round(avg_shap, 4),
                        "direction": "toward detractor" if avg_shap > 0 else "away from detractor",
                    })

            shap_details.append(detail)

        result["shap_details"] = shap_details

        # ---- SEGMENT TABLES (supporting detail) ----

        dimensions = [
            ("engagement_level", "Engagement Level"),
            ("transaction_role", "Transaction Role"),
            ("account_tenure_tier", "Account Tenure"),
            ("customer_lifecycle_stage", "Customer Lifecycle"),
            ("bidding_intensity_tier", "Bidding Intensity"),
            ("win_rate_tier", "Win Rate Tier"),
            ("avg_bid_value_tier", "Avg Bid Value Tier"),
            ("bid_recency", "Bid Recency"),
            ("region_name", "Region"),
            # nps_source excluded — covered in methodology note (survey channel bias)
        ]
        segments = [
            self._segment_analysis(
                df_fy.dropna(subset=[col]),
                df_all.dropna(subset=[col]),
                col, label
            )
            for col, label in dimensions if col in df_fy.columns
        ]
        # Sort: significant first (by p-value ascending), then non-significant
        segments.sort(key=lambda s: (
            0 if s["statistical_test"]["significant"] else 1,
            s["statistical_test"]["p_value"] if s["statistical_test"]["p_value"] is not None else 999
        ))
        result["segments"] = segments

        # Year-over-year shifts
        yoy_dims = ["engagement_level", "transaction_role", "account_tenure_tier"]
        yoy_results = []
        for col in yoy_dims:
            if col not in df_all.columns:
                continue
            for fy in df_all["nps_submitted_fiscal_year"].dropna().unique():
                fy_df = df_all[df_all["nps_submitted_fiscal_year"] == fy].dropna(subset=[col])
                for val, grp in fy_df.groupby(col):
                    total = len(grp)
                    det = (grp["nps_score_label"] == "detractor").sum()
                    pro = (grp["nps_score_label"] == "promoter").sum()
                    yoy_results.append({
                        "dimension": col, "segment": clean_label(str(val)),
                        "fiscal_year": int(fy), "count": int(total),
                        "detractor_pct": float(safe_pct(det, total)),
                        "nps": nps_from_counts(pro, det, total),
                    })
        result["year_over_year"] = yoy_results

        # Engagement paradox — FY26 display
        eng = df_fy.dropna(subset=["engagement_level"]).copy()
        eng["total_dollars"] = eng["bought_hammer_total"].fillna(0) + eng["sold_hammer_total"].fillna(0)
        paradox = []
        for level, grp in eng.groupby("engagement_level"):
            total = len(grp)
            det = (grp["nps_score_label"] == "detractor").sum()
            paradox.append({
                "engagement_level": clean_label(str(level)),
                "count": int(total),
                "detractor_pct": float(safe_pct(det, total)),
                "avg_total_dollars": float(round(grp["total_dollars"].mean(), 0)),
            })
        paradox.sort(key=lambda x: x["detractor_pct"], reverse=True)
        result["engagement_paradox"] = paradox

        # Bidding behavior comparison — FY26 display
        bid_cols = ["items_bid_on", "lifetime_bids", "wins", "win_rate_pct",
                    "avg_highest_bid", "serious_bids"]
        available = [c for c in bid_cols if c in df_fy.columns]
        bidding_comparison = []
        for label_val in ["detractor", "passive", "promoter"]:
            grp = df_fy[df_fy["nps_score_label"] == label_val]
            row = {"nps_label": label_val, "count": int(len(grp))}
            for col in available:
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                row[f"avg_{col}"] = round(float(vals.mean()), 2) if len(vals) > 0 else None
                row[f"median_{col}"] = round(float(vals.median()), 2) if len(vals) > 0 else None
            bidding_comparison.append(row)
        result["bidding_comparison"] = bidding_comparison

        # ---- LLM NARRATIVE (based on SHAP findings) ----
        result["narratives"] = self._generate_diagnostic_narratives(result)

        return result

    def _generate_diagnostic_narratives(self, diag_data: dict) -> dict:
        """Single DSPy call to narrate SHAP-based diagnostic findings."""
        print("    → Generating diagnostic narratives via LLM...")

        # Build SHAP summary for LLM
        lines = ["SHAP FEATURE IMPORTANCE (what drives detractor status, ranked):"]
        for i, feat in enumerate(diag_data["shap_importance"][:10], 1):
            lines.append(f"  {i}. {feat['feature']} (importance: {feat['mean_abs_shap']})")

        lines.append("\nSHAP VALUE DETAILS (how specific values push toward/away from detractor):")
        for detail in diag_data["shap_details"][:5]:
            impacts = ", ".join(
                f"{v['value']}({v['avg_shap']:+.3f} {'->det' if v['avg_shap'] > 0 else '<-det'}, n={v['count']})"
                for v in detail["value_impacts"][:5]
            )
            lines.append(f"  {detail['feature']}: {impacts}")

        # Add segment context for LLM awareness
        lines.append("\nSEGMENT BREAKDOWNS (FY26 supporting data):")
        for seg in diag_data["segments"]:
            sig = "SIG" if bool(seg["statistical_test"]["significant"]) else "ns"
            segs = ", ".join(f"{s['segment']}(NPS={s['nps']}, det={s['detractor_pct']}%)" for s in seg["segments"][:4])
            lines.append(f"  {seg['dimension']} [{sig}]: {segs}")

        # Engagement paradox
        para = ", ".join(f"{p['engagement_level']}(det={p['detractor_pct']}%, avg${p['avg_total_dollars']:,.0f})" for p in diag_data["engagement_paradox"])
        lines.append(f"\nENGAGEMENT PARADOX: {para}")

        model_q = diag_data["model_quality"]
        lines.append(f"\nMODEL QUALITY: XGBoost accuracy={model_q['cv_accuracy']:.1%} (baseline={model_q['baseline_accuracy']:.1%}), F1={model_q['cv_f1']:.1%}")

        shap_text = "\n".join(lines)

        # Collect SHAP feature names — top 5 only
        feat_names = [d["feature"] for d in diag_data["shap_details"][:5]]

        narrator = dspy.ChainOfThought(DiagnosticNarrative, max_tokens=4000)
        try:
            result = narrator(
                report_fy=REPORT_FY,
                nps_goal=NPS_GOAL,
                segment_data=shap_text,
                engagement_paradox=para,
                bidding_comparison="See SHAP details above",
                feature_list=", ".join(feat_names),
            )
            raw = result.narratives
            print(f"      Raw response length: {len(raw)} chars")
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            narratives = json.loads(raw.strip())
            print("      ✓ Diagnostic narratives generated")
            print(f"      Keys: {list(narratives.keys())}")
            if "feature_narratives" in narratives:
                print(f"      Feature narratives: {len(narratives['feature_narratives'])}")
                for feat, text in narratives["feature_narratives"].items():
                    print(f"        [feat: {feat}]: {text[:150]}")
            if "overall_synthesis" in narratives:
                print(f"      [SYNTHESIS]: {narratives['overall_synthesis'][:200]}")
            return narratives
        except Exception as e:
            import traceback
            print(f"      ✗ Narrative generation failed: {e}")
            traceback.print_exc()
            return {
                "overall_synthesis": "Diagnostic narrative unavailable.",
                "dimension_narratives": {}
            }


    # =========================================================================
    # LAYER 3: PREDICTIVE
    # =========================================================================

    def _build_predictive(self) -> dict:
        """Logistic regression risk model."""
        print("  [3/4] Building predictive model...")

        df = self._query_df(f"SELECT * FROM {TABLE}")
        result = {}

        df["is_detractor"] = (df["nps_score_label"] == "detractor").astype(int)

        cat_features = [
            "transaction_role", "account_tenure_tier", "engagement_level",
            "customer_lifecycle_stage", "bidding_intensity_tier",
            "win_rate_tier", "avg_bid_value_tier", "bid_recency",
        ]
        num_features = [
            "bought_item_count", "sold_item_count",
            "bought_hammer_total", "sold_hammer_total",
            "items_bid_on", "lifetime_bids", "wins",
            "win_rate_pct", "avg_highest_bid", "serious_bids",
            "account_tenure_days", "total_transaction_count",
        ]
        cat_features = [c for c in cat_features if c in df.columns]
        num_features = [c for c in num_features if c in df.columns]

        model_df = df[cat_features + num_features + ["is_detractor"]].copy()
        for col in cat_features:
            model_df[col] = model_df[col].fillna("unknown")
        for col in num_features:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0)

        encoders = {}
        for col in cat_features:
            le = LabelEncoder()
            model_df[col + "_enc"] = le.fit_transform(model_df[col])
            encoders[col] = le

        feature_cols = [c + "_enc" for c in cat_features] + num_features
        X = model_df[feature_cols].values
        y = model_df["is_detractor"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(lr, X_scaled, y, cv=cv, scoring="f1")

        result["cv_accuracy"] = round(float(cv_scores.mean()), 4)
        result["cv_accuracy_std"] = round(float(cv_scores.std()), 4)
        result["cv_f1"] = round(float(cv_f1.mean()), 4)

        lr.fit(X_scaled, y)

        feature_names = cat_features + num_features
        coefficients = lr.coef_[0]
        odds_ratios = np.exp(coefficients)

        importance = []
        for name, coef, odds in zip(feature_names, coefficients, odds_ratios):
            importance.append({
                "feature": name,
                "coefficient": round(float(coef), 4),
                "odds_ratio": round(float(odds), 4),
                "direction": "increases_risk" if coef > 0 else "decreases_risk",
                "abs_importance": round(abs(float(coef)), 4),
            })
        importance.sort(key=lambda x: x["abs_importance"], reverse=True)
        result["feature_importance"] = importance

        y_pred = lr.predict(X_scaled)
        cm = confusion_matrix(y, y_pred)
        result["confusion_matrix"] = {
            "true_negative": int(cm[0][0]), "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]), "true_positive": int(cm[1][1]),
        }

        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        result["classification_report"] = {
            "accuracy": round(report["accuracy"], 4),
            "detractor_precision": round(report.get("1", {}).get("precision", 0), 4),
            "detractor_recall": round(report.get("1", {}).get("recall", 0), 4),
            "detractor_f1": round(report.get("1", {}).get("f1-score", 0), 4),
        }

        baseline_accuracy = 1 - y.mean()
        result["baseline_accuracy"] = round(float(baseline_accuracy), 4)
        result["model_vs_baseline"] = round(float(cv_scores.mean() - baseline_accuracy), 4)

        result["model_interpretation"] = {
            "profile_predictive": result["cv_accuracy"] > baseline_accuracy + 0.05,
            "explanation": (
                "Customer profiles CAN predict detractor status — the model significantly outperforms baseline."
                if result["cv_accuracy"] > baseline_accuracy + 0.05
                else "NPS detraction appears EVENT-DRIVEN rather than PROFILE-DRIVEN. "
                     "The model barely outperforms baseline, meaning customer demographics and "
                     "transaction history alone cannot reliably predict who will be a detractor. "
                     "Specific experiences or service events likely trigger dissatisfaction."
            ),
        }

        return result

    # =========================================================================
    # LAYER 4: PRESCRIPTIVE
    # =========================================================================

    def _build_prescriptive(self, diagnostic: dict, predictive: dict) -> dict:
        """Data-driven recommendations from diagnostic + predictive findings."""
        print("  [4/4] Building prescriptive recommendations...")

        recs = []

        # Engagement paradox
        paradox = diagnostic.get("engagement_paradox", [])
        very_high = next((p for p in paradox if p["engagement_level"] == "very_high"), None)
        low = next((p for p in paradox if p["engagement_level"] == "low"), None)
        if very_high and low:
            recs.append({
                "priority": "critical",
                "category": "High-Value Customer Retention",
                "finding": (
                    f"Very high engagement customers have {very_high['detractor_pct']}% detractor rate "
                    f"vs {low['detractor_pct']}% for low engagement — "
                    f"a {round(very_high['detractor_pct'] - low['detractor_pct'], 1)}pp gap."
                ),
                "action": "Implement proactive account management for customers above $200k total transactions. "
                          "Assign dedicated reps, conduct quarterly satisfaction checks before NPS surveys.",
                "expected_impact": "Reducing very_high detractor rate by 5pp could save significant revenue from churn risk.",
            })

        # Transaction role
        segments = diagnostic.get("segments", [])
        role_seg = next((s for s in segments if s["group_column"] == "transaction_role"), None)
        if role_seg:
            seller_only = next((s for s in role_seg["segments"] if s["segment"] == "seller_only"), None)
            buyer_only = next((s for s in role_seg["segments"] if s["segment"] == "buyer_only"), None)
            if seller_only and buyer_only:
                recs.append({
                    "priority": "high",
                    "category": "Seller Experience",
                    "finding": (
                        f"Seller-only accounts are {seller_only['detractor_pct']}% detractors "
                        f"vs {buyer_only['detractor_pct']}% for buyer-only — "
                        f"sellers are {round(seller_only['detractor_pct'] / max(buyer_only['detractor_pct'], 0.1), 1)}x "
                        f"more likely to be detractors."
                    ),
                    "action": "Investigate seller-specific pain points: fee transparency, settlement timing, "
                              "lot condition reporting, buyer's premium communication. "
                              "Classify detractor comments from seller-only accounts to identify top themes.",
                    "expected_impact": "Seller satisfaction directly impacts consignment pipeline and repeat business.",
                })

        # Bidding behavior
        bid_comp = diagnostic.get("bidding_comparison", [])
        det_bid = next((b for b in bid_comp if b["nps_label"] == "detractor"), None)
        pro_bid = next((b for b in bid_comp if b["nps_label"] == "promoter"), None)
        if det_bid and pro_bid:
            det_wr = det_bid.get("avg_win_rate_pct")
            pro_wr = pro_bid.get("avg_win_rate_pct")
            if det_wr is not None and pro_wr is not None and det_wr < pro_wr:
                recs.append({
                    "priority": "medium",
                    "category": "Bidding Experience",
                    "finding": (
                        f"Detractors average {det_wr}% win rate vs {pro_wr}% for promoters. "
                        f"Losing more auctions correlates with lower satisfaction."
                    ),
                    "action": "Consider 'second chance' offers, bid coaching/alerts, or personalized "
                              "item recommendations to improve win rates for frequent losers.",
                    "expected_impact": "Improving bidder success rate addresses a root cause of dissatisfaction.",
                })

        # Model interpretation
        model_interp = predictive.get("model_interpretation", {})
        if not model_interp.get("profile_predictive", False):
            recs.append({
                "priority": "high",
                "category": "Measurement Approach",
                "finding": (
                    "The logistic regression model shows NPS detraction is event-driven, not profile-driven. "
                    "Customer demographics and history alone cannot predict who will be a detractor."
                ),
                "action": "Shift from profile-based risk scoring to event-based monitoring. "
                          "Track service touchpoints (title delays, condition disputes, fee surprises) "
                          "and flag accounts experiencing negative events for proactive outreach.",
                "expected_impact": "Event-based intervention catches dissatisfaction at the moment it happens, "
                                  "before it becomes an NPS detractor response.",
            })

        return {"recommendations": recs}

    # =========================================================================
    # CUSTOMER DRILL-DOWN (for dashboard tables)
    # =========================================================================

    def _build_drilldown(self) -> dict:
        """Individual customer records for dashboard drill-down tables."""
        print("  [+] Building customer drill-down data...")

        top_detractors = self._query_df(f"""
            SELECT entity_id, company_name, nps_score, nps_comment, nps_submitted_date,
                nps_submitted_fiscal_year as fy, transaction_role, engagement_level,
                account_tenure_tier, region_name, territory_name,
                bought_item_count, sold_item_count, bought_hammer_total, sold_hammer_total,
                items_bid_on, lifetime_bids, wins, win_rate_pct, avg_highest_bid
            FROM {TABLE}
            WHERE nps_score_label = 'detractor'
            ORDER BY (COALESCE(bought_hammer_total, 0) + COALESCE(sold_hammer_total, 0)) DESC
            LIMIT 50
        """)

        top_promoters = self._query_df(f"""
            SELECT entity_id, company_name, nps_score, nps_comment, nps_submitted_date,
                nps_submitted_fiscal_year as fy, transaction_role, engagement_level,
                account_tenure_tier, region_name, territory_name,
                bought_item_count, sold_item_count, bought_hammer_total, sold_hammer_total,
                items_bid_on, lifetime_bids, wins, win_rate_pct, avg_highest_bid
            FROM {TABLE}
            WHERE nps_score_label = 'promoter'
            ORDER BY (COALESCE(bought_hammer_total, 0) + COALESCE(sold_hammer_total, 0)) DESC
            LIMIT 50
        """)

        commented_detractors = self._query_df(f"""
            SELECT entity_id, company_name, nps_score, nps_comment, nps_submitted_date,
                nps_submitted_fiscal_year as fy, transaction_role, engagement_level,
                region_name, bought_hammer_total, sold_hammer_total
            FROM {TABLE}
            WHERE nps_score_label = 'detractor' AND has_comment = true
            ORDER BY (COALESCE(bought_hammer_total, 0) + COALESCE(sold_hammer_total, 0)) DESC
            LIMIT 100
        """)

        def to_records(df):
            return json.loads(df.to_json(orient="records", date_format="iso"))

        return {
            "top_detractors": to_records(top_detractors),
            "top_promoters": to_records(top_promoters),
            "commented_detractors": to_records(commented_detractors),
        }

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export(self, html_path: str = HTML_TEMPLATE):
        """Run all layers and inject JSON into the HTML template."""
        print(f"NPS Dashboard Data Builder")
        print(f"{'=' * 40}")
        print(f"Table: {TABLE}")
        print(f"HTML:  {html_path}\n")

        # Read the HTML template
        if not os.path.exists(html_path):
            print(f"ERROR: HTML template not found: {html_path}")
            sys.exit(1)

        with open(html_path, "r") as f:
            html_content = f.read()

        # Verify markers exist
        if DATA_START not in html_content or DATA_END not in html_content:
            print(f"ERROR: Data markers not found in {html_path}")
            print(f"  Expected: {DATA_START}...{DATA_END}")
            sys.exit(1)

        # Run all layers
        descriptive = self._build_descriptive()
        diagnostic = self._build_diagnostic()
        predictive = self._build_predictive()
        prescriptive = self._build_prescriptive(diagnostic, predictive)
        drilldown = self._build_drilldown()

        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "table": TABLE,
                "pipeline_version": "1.0.0",
            },
            "descriptive": descriptive,
            "diagnostic": diagnostic,
            "predictive": predictive,
            "prescriptive": prescriptive,
            "drilldown": drilldown,
        }

        # Serialize to JSON
        json_str = json.dumps(output, cls=JSONEncoder)

        # Inject into HTML: replace everything between markers
        start_idx = html_content.index(DATA_START)
        end_idx = html_content.index(DATA_END) + len(DATA_END)
        new_data = f"{DATA_START}{json_str}{DATA_END}"
        updated_html = html_content[:start_idx] + new_data + html_content[end_idx:]

        # Write back
        with open(html_path, "w") as f:
            f.write(updated_html)

        print(f"\n✓ Dashboard data injected into: {html_path}")
        print(f"  Report FY: {REPORT_FY} (goal: {NPS_GOAL})")
        print(f"  Descriptive: {len(descriptive['by_fiscal_year'])} fiscal years, {len(descriptive['quarterly_trend'])} quarters, {len(descriptive['comment_categories'])} comment summaries")
        print(f"  Diagnostic: {len(diagnostic['segments'])} dimensions analyzed")
        print(f"  Predictive: CV accuracy {predictive['cv_accuracy']:.1%} (baseline {predictive['baseline_accuracy']:.1%})")
        print(f"  Prescriptive: {len(prescriptive['recommendations'])} recommendations")
        print(f"  Drill-down: {len(drilldown['top_detractors'])} top detractors, {len(drilldown['commented_detractors'])} with comments")
        print(f"\n  Open in browser: file://{os.path.abspath(html_path)}")


# ============================================================
# STANDALONE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPS Dashboard Data Builder")
    parser.add_argument("--html", default=HTML_TEMPLATE, help=f"HTML template path (default: {HTML_TEMPLATE})")
    args = parser.parse_args()

    from shared.config import configure_llm
    configure_llm(temperature=0.3)

    db = AuctionDatabase()
    try:
        builder = NPSDashboardBuilder(db)
        builder.export(html_path=args.html)
    finally:
        db.close()