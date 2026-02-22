import dspy
from src.database import AuctionDatabase
from src.analyzer import AuctionAnalyzer

# Configure DSPy with Ollama
ollama_model = dspy.LM(
    model='ollama/llama3.1:8b',
    api_base='http://localhost:11434',
    api_key='ollama',
    max_tokens=500
)
dspy.configure(lm=ollama_model)

# Create database and agent
db = AuctionDatabase()
agent = AuctionAnalyzer(db)

print("=" * 60)
print("AUTONOMOUS AUCTION ANALYZER")
print("=" * 60)

# Test 1: Analyze single category
print("\n[TEST 1] Analyzing Heavy Equipment...")
print("-" * 60)
analysis = agent.analyze_single_category("Heavy Equipment")
print(analysis)

# Test 2: Compare categories
print("\n\n[TEST 2] Comparing Heavy Equipment vs Trucks...")
print("-" * 60)
comparison = agent.compare_two_categories("Heavy Equipment", "Trucks")
print(comparison)

db.close()
print("\n" + "=" * 60)
print("Analysis complete!")
