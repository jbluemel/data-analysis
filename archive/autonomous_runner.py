import schedule
import time
import dspy
from src.database import AuctionDatabase
from src.analyzer import AuctionAnalyzer
from src.report_generator import ReportGenerator
from datetime import datetime

def run_analysis():
    """This function runs on schedule"""
    print(f"\n{'='*60}")
    print(f"AUTONOMOUS ANALYSIS TRIGGERED: {datetime.now()}")
    print(f"{'='*60}\n")
    
    # Configure DSPy
    ollama_model = dspy.LM(
        model='ollama/llama3.1:8b',
        api_base='http://localhost:11434',
        api_key='ollama',
        max_tokens=500
    )
    dspy.configure(lm=ollama_model)
    
    # Run analysis
    db = AuctionDatabase()
    agent = AuctionAnalyzer(db)
    reporter = ReportGenerator(agent)
    
    report = reporter.generate_weekly_report()
    filepath = reporter.save_report(report)
    
    print(f"\n✓ Autonomous report complete: {filepath}\n")
    
    db.close()

# Schedule the job
schedule.every().monday.at("09:00").do(run_analysis)

# For testing: run every 1 minute
schedule.every(1).minutes.do(run_analysis)

print("="*60)
print("AUTONOMOUS AGENT RUNNING")
print("="*60)
print("Schedule:")
print("  - Every Monday at 9:00 AM (production)")
print("  - Every 1 minute (testing)")
print("\nPress Ctrl+C to stop")
print("="*60)

# Keep running
while True:
    schedule.run_pending()
    time.sleep(10)
