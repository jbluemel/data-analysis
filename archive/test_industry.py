import dspy
from src.database import AuctionDatabase
from src.analyzer import AuctionAnalyzer

ollama_model = dspy.LM(
    model='ollama/llama3.1:8b',
    api_base='http://localhost:11434',
    api_key='ollama',
    max_tokens=1000
)
dspy.configure(lm=ollama_model)

db = AuctionDatabase()
agent = AuctionAnalyzer(db)

print("=== Analyze by Industry ===")
result = agent.analyze_by_dimension('industry', fiscal_year=2026)
print(result)

db.close()
