"""
NPS Comment Topic Discovery
============================
Extracts topics from ALL NPS comments (promoter, passive, detractor),
then ranks them by frequency. No predefined categories — topics emerge
from the data.

Two-pass approach:
  Pass 1: LLM extracts raw topic labels from batches of comments
  Pass 2: LLM normalizes all unique labels into final categories
  Output: Ranked frequency table of final topics by NPS label

Usage:
    python -m analyzers.nps.extract_topics
"""

import json
import sys
import os
from collections import Counter

import dspy
import pandas as pd

from shared.database import AuctionDatabase
from shared.config import configure_llm


# ============================================================
# DSPY SIGNATURES
# ============================================================

class ExtractTopics(dspy.Signature):
    """Extract discussion topics from a batch of NPS survey comments for an equipment auction company.
    For each comment, identify 1-3 specific topic labels describing what the customer is talking about.
    
    RULES:
    - Be specific. "fees" is good. "user experience" is too vague.
    - Never use "general feedback", "overall experience", "user experience", or "platform" as topics.
    - Every comment is about SOMETHING specific — find it.
    - Good labels: fees, item descriptions, buyer premium, title delays, website navigation, customer service, 
      item condition, bidding process, shipping, payment speed, photos, inspection quality, account issues,
      inventory selection, selling process, communication, search functionality
    - Bad labels: general feedback, overall satisfaction, user experience, platform, positive experience
    
    Return ONLY valid JSON — no markdown, no preamble."""

    comments_batch: str = dspy.InputField(desc="Numbered list of NPS comments to analyze")
    topics: str = dspy.OutputField(desc="JSON object mapping comment number to list of specific topic strings, e.g. {\"1\": [\"fees\", \"item condition\"], \"2\": [\"website navigation\"]}")


class NormalizeTopics(dspy.Signature):
    """Given a list of raw topic labels extracted from equipment auction customer comments, 
    normalize them into final categories. Group only true synonyms together.
    
    RULES:
    - Keep categories specific and actionable — a specific team should be able to own each category
    - Maximum 2 words per category name
    - Maximum 15 final categories
    - NEVER create vague buckets like "user experience", "platform", "general feedback", "overall experience"
    - Only merge topics that mean the SAME thing (e.g. "buyer premium" + "10% fee" → "fees")
    - Do NOT merge topics handled by different teams (e.g. "website navigation" and "item descriptions" stay separate)
    - If a raw topic is already specific enough, keep it as-is
    
    Return ONLY valid JSON — no markdown, no preamble."""

    raw_topics: str = dspy.InputField(desc="Comma-separated list of all unique raw topic labels")
    normalized: str = dspy.OutputField(desc="JSON object mapping each raw topic to its final category, e.g. {\"buyer premium\": \"fees\", \"10% fee\": \"fees\", \"shipping delay\": \"logistics\"}")


# ============================================================
# MAIN
# ============================================================

def main():
    configure_llm(provider='claude', temperature=0.1)

    db = AuctionDatabase()
    print("NPS Comment Topic Discovery")
    print("=" * 50)

    # Pull all meaningful comments
    query = """
        SELECT entity_id, nps_score_label, nps_comment
        FROM nps_enriched
        WHERE nps_comment IS NOT NULL
          AND LENGTH(nps_comment) >= 20
        ORDER BY nps_score_label, entity_id
    """
    rows = db.query(query)
    df = pd.DataFrame(rows, columns=['entity_id', 'nps_score_label', 'nps_comment'])
    print(f"Loaded {len(df)} comments (20+ chars)")
    for label in ['detractor', 'passive', 'promoter']:
        count = len(df[df['nps_score_label'] == label])
        print(f"  {label}: {count}")

    # ---- PASS 1: Extract raw topics in batches ----
    print(f"\nPass 1: Extracting topics from {len(df)} comments...")
    batch_size = 25
    extractor = dspy.ChainOfThought(ExtractTopics, max_tokens=2000)

    all_comment_topics = []  # list of (entity_id, nps_score_label, [topics])

    for batch_start in range(0, len(df), batch_size):
        batch = df.iloc[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(df) + batch_size - 1) // batch_size

        # Build numbered comment list
        lines = []
        for idx, (_, row) in enumerate(batch.iterrows(), 1):
            comment = row['nps_comment'][:300]  # truncate long comments
            lines.append(f"{idx}. [{row['nps_score_label']}] {comment}")

        comments_text = "\n".join(lines)

        try:
            result = extractor(comments_batch=comments_text)
            raw = result.topics.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            topics_map = json.loads(raw.strip())

            for idx, (_, row) in enumerate(batch.iterrows(), 1):
                key = str(idx)
                topics = topics_map.get(key, ["general feedback"])
                # Normalize to lowercase
                topics = [t.strip().lower() for t in topics if t.strip()]
                all_comment_topics.append({
                    'entity_id': row['entity_id'],
                    'nps_score_label': row['nps_score_label'],
                    'topics': topics,
                })

            print(f"  Batch {batch_num}/{total_batches}: {len(batch)} comments processed")
        except Exception as e:
            print(f"  Batch {batch_num}/{total_batches}: FAILED - {e}")
            for _, row in batch.iterrows():
                all_comment_topics.append({
                    'entity_id': row['entity_id'],
                    'nps_score_label': row['nps_score_label'],
                    'topics': ['extraction_failed'],
                })

    # ---- Collect all unique raw topics ----
    raw_counter = Counter()
    for item in all_comment_topics:
        for topic in item['topics']:
            raw_counter[topic] += 1

    print(f"\nPass 1 complete: {len(raw_counter)} unique raw topics found")
    print("Top 30 raw topics:")
    for topic, count in raw_counter.most_common(30):
        print(f"  {count:4d}  {topic}")

    # ---- PASS 2: Normalize topics ----
    print(f"\nPass 2: Normalizing {len(raw_counter)} raw topics into final categories...")
    normalizer = dspy.ChainOfThought(NormalizeTopics, max_tokens=3000)

    raw_list = ", ".join(sorted(raw_counter.keys()))
    try:
        result = normalizer(raw_topics=raw_list)
        raw = result.normalized.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        norm_map = json.loads(raw.strip())
        print(f"  Normalized into {len(set(norm_map.values()))} final categories")
    except Exception as e:
        print(f"  Normalization failed: {e}")
        print("  Falling back to raw topics")
        norm_map = {t: t for t in raw_counter.keys()}

    # ---- Apply normalization and build final counts ----
    final_counts = {'all': Counter(), 'detractor': Counter(), 'passive': Counter(), 'promoter': Counter()}

    for item in all_comment_topics:
        label = item['nps_score_label']
        for raw_topic in item['topics']:
            final = norm_map.get(raw_topic, raw_topic)
            final_counts['all'][final] += 1
            final_counts[label][final] += 1

    # ---- Output ----
    print("\n" + "=" * 70)
    print("TOPIC FREQUENCY RANKING")
    print("=" * 70)
    print(f"\n{'TOPIC':<30} {'TOTAL':>6} {'DET':>6} {'PAS':>6} {'PRO':>6}")
    print("-" * 70)

    for topic, total in final_counts['all'].most_common(30):
        det = final_counts['detractor'].get(topic, 0)
        pas = final_counts['passive'].get(topic, 0)
        pro = final_counts['promoter'].get(topic, 0)
        print(f"{topic:<30} {total:>6} {det:>6} {pas:>6} {pro:>6}")

    # ---- Save normalization map for reference ----
    output = {
        'raw_topic_counts': dict(raw_counter.most_common()),
        'normalization_map': norm_map,
        'final_counts': {
            'all': dict(final_counts['all'].most_common()),
            'detractor': dict(final_counts['detractor'].most_common()),
            'passive': dict(final_counts['passive'].most_common()),
            'promoter': dict(final_counts['promoter'].most_common()),
        },
        'comment_topics': all_comment_topics,
    }

    output_path = "analyzers/nps/topic_discovery_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_path}")

    # ---- Summary ----
    n_final = len(set(norm_map.values()))
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Comments processed: {len(all_comment_topics)}")
    print(f"  Raw topics found:   {len(raw_counter)}")
    print(f"  Final topics:   {n_final}")
    print(f"\nUse these rankings to select the final 6-8 topic categories")
    print(f"for the nps_enriched gold table update.")


if __name__ == "__main__":
    main()