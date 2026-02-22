from src.database import AuctionDatabase

# Create database connection
db = AuctionDatabase()

print("Testing database connection...")

# Test 1: Get categories
print("\n1. Available categories:")
categories = db.get_all_categories()
print(f"   {categories}")

# Test 2: Get Heavy Equipment stats
print("\n2. Heavy Equipment stats:")
stats = db.get_category_stats("Heavy Equipment")
print(f"   Count: {stats['count']}")
print(f"   Avg Price: ${stats['avg_price']:,.2f}")
print(f"   Price Range: ${stats['min_price']:,.2f} - ${stats['max_price']:,.2f}")

# Test 3: Get a few items
print("\n3. Sample Heavy Equipment items:")
items = db.get_items(category="Heavy Equipment", limit=3)
for item in items:
    print(f"   {item['unique_id']}: {item['model']} - ${item['hammer']:,.2f}")

db.close()
print("\n✓ Database connection working!")
