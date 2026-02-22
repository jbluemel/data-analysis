import dspy
from src.database import AuctionDatabase
from src.analyzer import AuctionAnalyzer

# Configure DSPy with local Llama
ollama_model = dspy.LM(
    model='ollama/llama3.1:8b',
    api_base='http://localhost:11434',
    api_key='ollama',
    max_tokens=1000
)
dspy.configure(lm=ollama_model)

# Create analyzer
db = AuctionDatabase()
agent = AuctionAnalyzer(db)

# Test 1: Check available tables
print("=== Available Tables ===")
for t in agent.get_tables():
    print(f"  - {t}")

# Test 2: Analyze weekly summary
print("\n=== Weekly Summary Analysis ===")
result = agent.analyze_weekly_summary(fiscal_year=2026)
print(result)

db.close()
print("\n✓ Test complete!")
