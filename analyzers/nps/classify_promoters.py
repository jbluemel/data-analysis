"""
NPS Promoter Comment Classification
======================================
Tags each promoter comment against 8 predefined praise categories.
A comment can mention multiple categories.

Categories:
  1. customer_service - great service, helpful staff, responsive reps, good communication
  2. inventory_selection - variety of equipment, lots to choose from, good selection
  3. website_ease - easy to navigate, user friendly, good search, site design
  4. pricing_deals - good prices, fair deals, affordable, value for money
  5. bidding_experience - easy to bid, smooth auction process, straightforward
  6. selling_experience - great platform to sell, reaches many buyers, good returns
  7. trust_reliability - legitimate site, reliable, honest, no reserve policy works
  8. item_information - good descriptions, thorough photos/videos, accurate listings

Usage:
    python -m analyzers.nps.classify_promoters
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
    "customer_service",
    "inventory_selection",
    "website_ease",
    "pricing_deals",
    "bidding_experience",
    "selling_experience",
    "trust_reliability",
    "item_information",
]


class ClassifyComment(dspy.Signature):
    """Classify a batch of promoter comments from an equipment auction company.
    For each comment, determine which praise categories it mentions.
    
    Categories:
    - customer_service: great service, helpful staff, responsive reps, good people, good communication
    - inventory_selection: variety of equipment, lots to choose from, good selection, wide range
    - website_ease: easy to navigate, user friendly, good search, nice site, simple to use
    - pricing_deals: good prices, fair deals, affordable, great value, competitive pricing
    - bidding_experience: easy to bid, smooth auction process, straightforward bidding, fun to bid
    - selling_experience: great platform to sell, reaches many buyers, good returns, easy to list
    - trust_reliability: legitimate site, reliable, honest company, trustworthy, no reserve works
    - item_information: good descriptions, thorough photos/videos, accurate listings, detailed info
    
    A comment can match multiple categories or none.
    Return ONLY valid JSON — no markdown, no preamble."""

    comments_batch: str = dspy.InputField(desc="Numbered list of promoter comments")
    classifications: str = dspy.OutputField(desc='JSON object mapping comment number to list of matched categories, e.g. {"1": ["customer_service", "website_ease"], "2": ["inventory_selection"], "3": []}')


def main():
    configure_llm(provider='claude', temperature=0.1)

    db = AuctionDatabase()
    print("NPS Promoter Comment Classification")
    print("=" * 60)

    # Pull promoter comments only
    query = """
        SELECT entity_id, nps_score_label, nps_comment
        FROM nps_enriched
        WHERE nps_score_label = 'promoter'
          AND nps_comment IS NOT NULL
          AND LENGTH(nps_comment) >= 20
        ORDER BY entity_id
    """
    rows = db.query(query)
    df = pd.DataFrame(rows, columns=['entity_id', 'nps_score_label', 'nps_comment'])
    print(f"Loaded {len(df)} promoter comments")

    # ---- Classify in batches ----
    batch_size = 15
    classifier = dspy.ChainOfThought(ClassifyComment, max_tokens=2000)
    
    all_results = []

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
    print(f"PROMOTER PRAISE CATEGORIES")
    print(f"{'=' * 60}")
    print(f"\n{'CATEGORY':<25} {'COUNT':>6} {'% OF PROMOTERS':>15}")
    print("-" * 50)

    for cat, count in cat_counter.most_common():
        pct = round(count / len(df) * 100, 1)
        print(f"{cat:<25} {count:>6} {pct:>14.1f}%")

    print(f"\n  Total promoters: {len(df)}")
    print(f"  No category matched: {no_match}")
    print(f"  Multiple categories: {multi_match}")

    # ---- Show sample comments per category ----
    print(f"\n{'=' * 60}")
    print(f"SAMPLE COMMENTS PER CATEGORY")
    print(f"{'=' * 60}")

    comment_lookup = {int(row['entity_id']): row['nps_comment'] for _, row in df.iterrows()}

    for cat, _ in cat_counter.most_common():
        matching = [r for r in all_results if cat in r['categories']]
        print(f"\n--- {cat.upper()} ({len(matching)} comments) ---")
        for item in matching[:3]:
            comment = comment_lookup.get(item['entity_id'], 'N/A')[:150]
            print(f"  • {comment}...")

    # ---- Save results ----
    output = {
        'total_promoters': len(df),
        'no_match': no_match,
        'multi_match': multi_match,
        'category_counts': dict(cat_counter.most_common()),
        'classifications': all_results,
    }

    output_path = "analyzers/nps/promoter_classifications.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()