"""
Update nps_enriched with Topic Columns
========================================
Reads detractor and promoter classification JSONs,
adds 16 topic columns to nps_enriched table.

All columns are single-direction:
  Detractor columns: -1 when mentioned, 0 otherwise
  Promoter columns:  +1 when mentioned, 0 otherwise

Usage:
    python -m analyzers.nps.update_topics_gold
"""

import json
import psycopg2

from shared.database import AuctionDatabase


# Detractor categories → column mapping (value = -1)
DETRACTOR_COLS = {
    "fees": "topic_fees",
    "item_accuracy": "topic_item_accuracy",
    "item_condition": "topic_item_condition",
    "bidding_issues": "topic_auction_mechanics",
    "account_problems": "topic_account",
    "seller_honesty": "topic_seller_honesty",
    "payment_settlement": "topic_payment",
    "title_paperwork": "topic_title_paperwork",
}

# Promoter categories → column mapping (value = +1)
PROMOTER_COLS = {
    "customer_service": "topic_customer_service",
    "inventory_selection": "topic_inventory",
    "website_ease": "topic_website",
    "pricing_deals": "topic_pricing_deals",
    "bidding_experience": "topic_bidding_ease",
    "selling_experience": "topic_selling",
    "trust_reliability": "topic_trust",
    "item_information": "topic_item_info",
}

# All unique column names
ALL_TOPIC_COLS = sorted(set(list(DETRACTOR_COLS.values()) + list(PROMOTER_COLS.values())))


def main():
    print("Updating nps_enriched with topic columns")
    print("=" * 60)

    # ---- Load classification results ----
    with open("analyzers/nps/detractor_classifications.json") as f:
        det_data = json.load(f)
    with open("analyzers/nps/promoter_classifications.json") as f:
        pro_data = json.load(f)

    print(f"Loaded {len(det_data['classifications'])} detractor classifications")
    print(f"Loaded {len(pro_data['classifications'])} promoter classifications")

    # ---- Build entity_id → topic values map ----
    entity_topics = {}

    # Detractor mentions → -1
    for item in det_data['classifications']:
        eid = item['entity_id']
        if eid not in entity_topics:
            entity_topics[eid] = {col: 0 for col in ALL_TOPIC_COLS}
        for cat in item['categories']:
            if cat in DETRACTOR_COLS:
                col = DETRACTOR_COLS[cat]
                entity_topics[eid][col] = -1

    # Promoter mentions → +1
    for item in pro_data['classifications']:
        eid = item['entity_id']
        if eid not in entity_topics:
            entity_topics[eid] = {col: 0 for col in ALL_TOPIC_COLS}
        for cat in item['categories']:
            if cat in PROMOTER_COLS:
                col = PROMOTER_COLS[cat]
                entity_topics[eid][col] = 1

    print(f"Entity IDs with topics: {len(entity_topics)}")

    # ---- Connect to database ----
    db = AuctionDatabase()
    conn = db.conn
    cur = conn.cursor()

    # ---- Add columns if they don't exist ----
    print(f"\nAdding {len(ALL_TOPIC_COLS)} topic columns to nps_enriched...")
    for col in ALL_TOPIC_COLS:
        try:
            cur.execute(f"ALTER TABLE nps_enriched ADD COLUMN {col} INTEGER DEFAULT 0")
            print(f"  Added: {col}")
        except psycopg2.errors.DuplicateColumn:
            conn.rollback()
            print(f"  Exists: {col}")
    conn.commit()

    # ---- Set all topic columns to 0 first (in case of re-run) ----
    set_clause = ", ".join([f"{col} = 0" for col in ALL_TOPIC_COLS])
    cur.execute(f"UPDATE nps_enriched SET {set_clause}")
    reset_count = cur.rowcount
    conn.commit()
    print(f"\nReset {reset_count} rows to 0")

    # ---- Update rows with topic values ----
    updated = 0
    for eid, topics in entity_topics.items():
        non_zero = {col: val for col, val in topics.items() if val != 0}
        if not non_zero:
            continue

        set_clause = ", ".join([f"{col} = %s" for col in non_zero.keys()])
        values = list(non_zero.values()) + [eid]
        cur.execute(
            f"UPDATE nps_enriched SET {set_clause} WHERE entity_id = %s",
            values
        )
        updated += 1

    conn.commit()
    print(f"Updated {updated} rows with topic values")

    # ---- Verify: counts by label ----
    print(f"\n{'=' * 60}")
    print("VERIFICATION — TOPIC COUNTS BY NPS LABEL")
    print(f"{'=' * 60}")

    # Detractor topics
    print(f"\nDETRACTOR TOPICS (should only appear on detractors):")
    print(f"{'COLUMN':<28} {'DET':>5} {'PAS':>5} {'PRO':>5}")
    print("-" * 48)
    for col in sorted(DETRACTOR_COLS.values()):
        cur.execute(f"""
            SELECT 
                SUM(CASE WHEN nps_score_label = 'detractor' AND {col} = -1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN nps_score_label = 'passive' AND {col} = -1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN nps_score_label = 'promoter' AND {col} = -1 THEN 1 ELSE 0 END)
            FROM nps_enriched
        """)
        row = cur.fetchone()
        print(f"{col:<28} {row[0]:>5} {row[1]:>5} {row[2]:>5}")

    # Promoter topics
    print(f"\nPROMOTER TOPICS (should only appear on promoters):")
    print(f"{'COLUMN':<28} {'DET':>5} {'PAS':>5} {'PRO':>5}")
    print("-" * 48)
    for col in sorted(PROMOTER_COLS.values()):
        cur.execute(f"""
            SELECT 
                SUM(CASE WHEN nps_score_label = 'detractor' AND {col} = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN nps_score_label = 'passive' AND {col} = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN nps_score_label = 'promoter' AND {col} = 1 THEN 1 ELSE 0 END)
            FROM nps_enriched
        """)
        row = cur.fetchone()
        print(f"{col:<28} {row[0]:>5} {row[1]:>5} {row[2]:>5}")

    # Summary
    cur.execute(f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN {' + '.join([f'ABS({c})' for c in ALL_TOPIC_COLS])} > 0 THEN 1 ELSE 0 END) as has_any_topic
        FROM nps_enriched
    """)
    row = cur.fetchone()
    print(f"\nTotal rows: {row[0]}")
    print(f"Rows with at least one topic: {row[1]}")
    print(f"Rows with no topics: {row[0] - row[1]}")

    cur.close()
    print(f"\nDone. nps_enriched now has {len(ALL_TOPIC_COLS)} topic columns.")


if __name__ == "__main__":
    main()