import dspy
from config import configure_llm
from src.database import AuctionDatabase
from src.analyzer import AuctionAnalyzer

# Configure Claude
configure_llm(provider='claude', temperature=0.3)

# Create analyzer
db = AuctionDatabase()
agent = AuctionAnalyzer(db)

# Test explain_week on FY26 Week 22 (the most recent week)
print("="*60)
print("TESTING SCORECARD-BASED WEEK EXPLANATION")
print("="*60)
print()

explanation = agent.explain_week(fiscal_year='FY 2026', fiscal_week=22)

print(explanation)
print()
print("="*60)

db.close()
