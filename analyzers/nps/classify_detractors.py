"""
NPS Detractor Comment Classification
======================================
Tags each detractor comment against 8 predefined complaint categories.
Each comment gets a sentiment-scored tag: -1 (negative), 0 (not mentioned), +1 (positive).
A comment can mention multiple categories.

Categories:
  1. fees - buyer premium, commission, 10% fee, listing fees, seller costs
  2. item_accuracy - descriptions wrong, photos/videos misleading, listing misrepresentation
  3. item_condition - equipment broken, undisclosed defects, not as expected
  4. title_paperwork - title delays, tax issues, documentation problems
  5. seller_honesty - seller lied, backed out, Purple Wave didn't back buyer
  6. payment_settlement - slow payouts, invoicing issues, credit/payment problems
  7. bidding_issues - accidental bids, can't cancel, auction mechanics frustration
  8. account_problems - registration, verification, suspended accounts, login issues

Usage:
    python -m analyzers.nps.classify_detractors
"""

import json
import sys
import os
from collections import Counter

import dspy
import pandas as pd

from shared.database import AuctionDatabase
from shared.config import configure_llm


CATEGORIES = [
    "fees",
    "item_accuracy",
    "item_condition",
    "title_paperwork",
    "seller_honesty",
    "payment_settlement",
    "bidding_issues",
    "account_problems",
]


class ClassifyComment(dspy.Signature):
    """Classify a batch of detractor comments from an equipment auction company.
    For each comment, determine which complaint categories it mentions.
    
    Categories:
    - fees: buyer premium, commission, 10% fee, listing fees, seller costs, pricing complaints
    - item_accuracy: descriptions wrong, photos/videos misleading, listing misrepresentation, poor inspection writeup
    - item_condition: equipment broken, undisclosed defects, item not as expected, hidden damage
    - title_paperwork: title delays, tax issues, documentation problems, slow paperwork processing
    - seller_honesty: seller lied, seller backed out, Purple Wave didn't protect buyer, fraud concerns
    - payment_settlement: slow payouts to sellers, invoicing issues, credit problems, payment method frustration
    - bidding_issues: accidental bids can't be cancelled, auction timing, bid manipulation concerns, bidding mechanics
    - account_problems: registration trouble, phone verification, suspended account, login issues
    
    A comment can match multiple categories or none.
    Return ONLY valid JSON — no markdown, no preamble."""

    comments_batch: str = dspy.InputField(desc="Numbered list of detractor comments")
    classifications: str = dspy.OutputField(desc='JSON object mapping comment number to list of matched categories, e.g. {"1": ["fees", "item_condition"], "2": ["bidding_issues"], "3": []}')


def main():
    configure_llm(provider='claude', temperature=0.1)

    db = AuctionDatabase()
    print("NPS Detractor Comment Classification")
    print("=" * 60)

    # Pull detractor comments only
    query = """
        SELECT entity_id, nps_score_label, nps_comment
        FROM nps_enriched
        WHERE nps_score_label = 'detractor'
          AND nps_comment IS NOT NULL
          AND LENGTH(nps_comment) >= 20
        ORDER BY entity_id
    """
    rows = db.query(query)
    df = pd.DataFrame(rows, columns=['entity_id', 'nps_score_label', 'nps_comment'])
    print(f"Loaded {len(df)} detractor comments")

    # ---- Classify in batches ----
    batch_size = 15
    classifier = dspy.ChainOfThought(ClassifyComment, max_tokens=2000)
    
    all_results = []  # list of (entity_id, [categories])

    for batch_start in range(0, len(df), batch_size):
        batch = df.iloc[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(df) + batch_size - 1) // batch_size

        lines = []
        for idx, (_, row) in enumerate(batch.iterrows(), 1):
            comment = row['nps_comment'][:400]
            lines.append(f"{idx}. {comment}")

        comments_text = "\n".join(lines)

        try:
            result = classifier(comments_batch=comments_text)
            raw = result.classifications.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            class_map = json.loads(raw.strip())

            for idx, (_, row) in enumerate(batch.iterrows(), 1):
                key = str(idx)
                cats = class_map.get(key, [])
                # Validate categories
                cats = [c.strip().lower() for c in cats if c.strip().lower() in CATEGORIES]
                all_results.append({
                    'entity_id': int(row['entity_id']),
                    'categories': cats,
                })

            print(f"  Batch {batch_num}/{total_batches}: {len(batch)} comments classified")
        except Exception as e:
            print(f"  Batch {batch_num}/{total_batches}: FAILED - {e}")
            for _, row in batch.iterrows():
                all_results.append({
                    'entity_id': int(row['entity_id']),
                    'categories': ['classification_failed'],
                })

    # ---- Count results ----
    cat_counter = Counter()
    no_match = 0
    multi_match = 0

    for item in all_results:
        if len(item['categories']) == 0:
            no_match += 1
        if len(item['categories']) > 1:
            multi_match += 1
        for cat in item['categories']:
            cat_counter[cat] += 1

    # ---- Output ----
    print(f"\n{'=' * 60}")
    print(f"DETRACTOR COMPLAINT CATEGORIES")
    print(f"{'=' * 60}")
    print(f"\n{'CATEGORY':<25} {'COUNT':>6} {'% OF DETRACTORS':>15}")
    print("-" * 50)

    for cat, count in cat_counter.most_common():
        pct = round(count / len(df) * 100, 1)
        print(f"{cat:<25} {count:>6} {pct:>14.1f}%")

    print(f"\n  Total detractors: {len(df)}")
    print(f"  No category matched: {no_match}")
    print(f"  Multiple categories: {multi_match}")

    # ---- Show sample comments per category ----
    print(f"\n{'=' * 60}")
    print(f"SAMPLE COMMENTS PER CATEGORY")
    print(f"{'=' * 60}")

    # Build entity_id -> comment lookup
    comment_lookup = {int(row['entity_id']): row['nps_comment'] for _, row in df.iterrows()}

    for cat, _ in cat_counter.most_common():
        matching = [r for r in all_results if cat in r['categories']]
        print(f"\n--- {cat.upper()} ({len(matching)} comments) ---")
        for item in matching[:3]:
            comment = comment_lookup.get(item['entity_id'], 'N/A')[:150]
            print(f"  • {comment}...")

    # ---- Save results ----
    output = {
        'total_detractors': len(df),
        'no_match': no_match,
        'multi_match': multi_match,
        'category_counts': dict(cat_counter.most_common()),
        'classifications': all_results,
    }

    output_path = "analyzers/nps/detractor_classifications.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()