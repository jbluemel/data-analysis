import dspy
from config import configure_llm
from src.database import AuctionDatabase
from src.analyzer import AuctionAnalyzer

# Configure - change these to experiment
configure_llm(provider='claude', temperature=0.3)

# Create analyzer
db = AuctionDatabase()
agent = AuctionAnalyzer(db)

# Run analyses
print("\n" + "="*60)
print("WEEKLY SUMMARY")
print("="*60)
print(agent.analyze_weekly_summary(fiscal_year=2026))

print("\n" + "="*60)
print("BY INDUSTRY")
print("="*60)
print(agent.analyze_by_dimension('industry', fiscal_year=2026))

print("\n" + "="*60)
print("PROMPT SENT TO CLAUDE")
print("="*60)
dspy.inspect_history(n=1)

db.close()
