"""
NPS Comment Topic Discovery — Embedding-Based Clustering
==========================================================
Uses BERTopic to find natural topic clusters from NPS comments.
No predefined categories — topics emerge from comment similarity.

Pipeline:
  1. Load all meaningful comments from nps_enriched
  2. Generate sentence embeddings (all-MiniLM-L6-v2)
  3. Reduce dimensions (UMAP)
  4. Cluster similar comments (HDBSCAN)
  5. Extract topic labels (c-TF-IDF)
  6. Display clusters with representative comments and NPS label breakdown

Usage:
    python -m analyzers.nps.cluster_topics
"""

import json
import sys
import os
from collections import Counter

import pandas as pd
import numpy as np

from shared.database import AuctionDatabase


def main():
    db = AuctionDatabase()
    print("NPS Comment Topic Discovery — Embedding Clustering")
    print("=" * 60)

    # ---- Load comments ----
    query = """
        SELECT entity_id, nps_score_label, nps_comment
        FROM nps_enriched
        WHERE nps_comment IS NOT NULL
          AND LENGTH(nps_comment) >= 20
        ORDER BY entity_id
    """
    rows = db.query(query)
    df = pd.DataFrame(rows, columns=['entity_id', 'nps_score_label', 'nps_comment'])
    print(f"Loaded {len(df)} comments (20+ chars)")
    for label in ['detractor', 'passive', 'promoter']:
        print(f"  {label}: {len(df[df['nps_score_label'] == label])}")

    comments = df['nps_comment'].tolist()

    # ---- BERTopic clustering with TF-IDF (no model download needed) ----
    print(f"\nRunning BERTopic clustering...")
    print("  Step 1: Generating TF-IDF embeddings (local, no download)...")

    from bertopic import BERTopic
    from sklearn.feature_extraction.text import TfidfVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    # Use TF-IDF instead of sentence-transformers (avoids HuggingFace download)
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),      # unigrams + bigrams
        stop_words='english',
        min_df=3,                # must appear in at least 3 comments
    )
    embeddings = tfidf.fit_transform(comments).toarray()

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=20,     # minimum comments per topic
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics="auto",
        top_n_words=10,
        verbose=True,
    )

    print("  Step 2: Fitting model (reduce → cluster)...")
    topics, probs = topic_model.fit_transform(comments, embeddings=embeddings)

    # ---- Results ----
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info['Topic'] != -1])
    n_outliers = len(topic_info[topic_info['Topic'] == -1])

    print(f"\n{'=' * 70}")
    print(f"CLUSTERING RESULTS")
    print(f"{'=' * 70}")
    print(f"  Topics found: {n_topics}")
    outlier_count = int(topic_info[topic_info['Topic'] == -1]['Count'].values[0]) if n_outliers > 0 else 0
    print(f"  Outliers (no cluster): {outlier_count} comments")
    print(f"  Clustered: {len(df) - outlier_count} comments")

    # Add topic assignments to dataframe
    df['topic_id'] = topics

    # ---- Display each topic ----
    print(f"\n{'=' * 70}")
    print(f"TOPIC DETAILS")
    print(f"{'=' * 70}")

    topic_summary = []

    for _, row in topic_info.iterrows():
        tid = row['Topic']
        if tid == -1:
            continue  # skip outliers for now

        count = row['Count']
        # Get top words for this topic
        topic_words = topic_model.get_topic(tid)
        label_words = [w for w, _ in topic_words[:8]]

        # Get NPS breakdown
        topic_df = df[df['topic_id'] == tid]
        det = len(topic_df[topic_df['nps_score_label'] == 'detractor'])
        pas = len(topic_df[topic_df['nps_score_label'] == 'passive'])
        pro = len(topic_df[topic_df['nps_score_label'] == 'promoter'])
        det_pct = round(det / count * 100, 1) if count > 0 else 0

        # Get representative comments
        rep_docs = topic_model.get_representative_docs(tid)

        print(f"\n--- Topic {tid} ({count} comments, {det_pct}% detractor) ---")
        print(f"  Keywords: {', '.join(label_words)}")
        print(f"  NPS: {det} det / {pas} pas / {pro} pro")
        print(f"  Sample comments:")
        if rep_docs:
            for doc in rep_docs[:3]:
                preview = doc[:150].replace('\n', ' ')
                print(f"    • {preview}...")

        topic_summary.append({
            'topic_id': tid,
            'count': count,
            'keywords': label_words,
            'detractor': det,
            'passive': pas,
            'promoter': pro,
            'detractor_pct': det_pct,
            'sample_comments': rep_docs[:3] if rep_docs else [],
        })

    # ---- Outlier info ----
    if outlier_count > 0:
        outlier_df = df[df['topic_id'] == -1]
        det = len(outlier_df[outlier_df['nps_score_label'] == 'detractor'])
        pas = len(outlier_df[outlier_df['nps_score_label'] == 'passive'])
        pro = len(outlier_df[outlier_df['nps_score_label'] == 'promoter'])
        print(f"\n--- Outliers / Unclustered ({outlier_count} comments) ---")
        print(f"  NPS: {det} det / {pas} pas / {pro} pro")
        print(f"  Sample:")
        for _, r in outlier_df.head(3).iterrows():
            preview = r['nps_comment'][:150].replace('\n', ' ')
            print(f"    • [{r['nps_score_label']}] {preview}...")

    # ---- Ranked summary table ----
    print(f"\n{'=' * 70}")
    print(f"RANKED BY DETRACTOR CONCENTRATION")
    print(f"{'=' * 70}")
    print(f"{'TOPIC':>5} {'KEYWORDS':<45} {'TOTAL':>5} {'DET':>5} {'DET%':>5}")
    print("-" * 70)

    ranked = sorted(topic_summary, key=lambda x: x['detractor_pct'], reverse=True)
    for t in ranked:
        kw = ', '.join(t['keywords'][:5])
        print(f"{t['topic_id']:>5} {kw:<45} {t['count']:>5} {t['detractor']:>5} {t['detractor_pct']:>5.1f}")

    # ---- Save results ----
    output = {
        'n_topics': n_topics,
        'outlier_count': outlier_count,
        'topics': topic_summary,
        'comment_assignments': [
            {'entity_id': int(r['entity_id']), 'nps_score_label': r['nps_score_label'], 'topic_id': int(r['topic_id'])}
            for _, r in df.iterrows()
        ],
    }

    output_path = "analyzers/nps/cluster_topic_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    print(f"\nNext step: Review topics, name them, then update nps_enriched gold table.")


if __name__ == "__main__":
    main()