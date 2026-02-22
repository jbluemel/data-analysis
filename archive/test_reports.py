import dspy
from src.database import AuctionDatabase
from src.analyzer import AuctionAnalyzer
from src.report_generator import ReportGenerator

# Configure DSPy
ollama_model = dspy.LM(
    model='ollama/llama3.1:8b',
    api_base='http://localhost:11434',
    api_key='ollama',
    max_tokens=500
)
dspy.configure(lm=ollama_model)

# Create components
db = AuctionDatabase()
agent = AuctionAnalyzer(db)
reporter = ReportGenerator(agent)

# Generate and save report
print("=" * 60)
print("GENERATING WEEKLY REPORT")
print("=" * 60)

report = reporter.generate_weekly_report()
filepath = reporter.save_report(report)

print("\n" + "=" * 60)
print(f"Report complete! Saved to: {filepath}")
print("=" * 60)

db.close()
