from src.database import AuctionDatabase

db = AuctionDatabase()

# Test 1: List tables
print("=== Available Tables ===")
for table in db.list_tables():
    print(f"  - {table}")

# Test 2: Describe a table
print("\n=== weekly_metrics_summary schema ===")
for col in db.describe_table('weekly_metrics_summary'):
    print(f"  {col['column_name']}: {col['data_type']}")

# Test 3: Run a query
print("\n=== Query: FY26 weekly summary ===")
results = db.query("SELECT fiscal_week_number, total_items_sold, avg_lot_value FROM weekly_metrics_summary ORDER BY fiscal_week_number")
for row in results:
    print(f"  Week {int(row['fiscal_week_number'])}: {row['total_items_sold']} items, ${row['avg_lot_value']:,.0f} avg")

# Test 4: Query a dimensional table
print("\n=== Query: Top 5 industries by volume ===")
results = db.query("""
    SELECT taxonomy_industry, SUM(total_items_sold) as total_items 
    FROM weekly_metrics_by_industry 
    GROUP BY taxonomy_industry 
    ORDER BY total_items DESC 
    LIMIT 5
""")
for row in results:
    print(f"  {row['taxonomy_industry']}: {row['total_items']} items")

db.close()
print("\n✓ All tests passed!")
