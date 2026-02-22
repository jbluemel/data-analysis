import os
import dspy
from src.database import AuctionDatabase
from src.analyzer import AuctionAnalyzer

# Configure DSPy with Claude
claude_model = dspy.LM(
    model='anthropic/claude-sonnet-4-20250514',
    api_key=os.environ.get('ANTHROPIC_API_KEY'),
    max_tokens=1000
)
dspy.configure(lm=claude_model)

db = AuctionDatabase()
agent = AuctionAnalyzer(db)

print("=== Analyze by Industry (Claude) ===")
result = agent.analyze_by_dimension('industry', fiscal_year=2026)
print(result)

db.close()
print("\n✓ Claude test complete!")
