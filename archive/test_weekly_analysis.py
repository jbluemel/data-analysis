#!/usr/bin/env python3
"""
Test script for weekly trend analysis
"""
import dspy
from src.analyzer import AuctionAnalyzer
from src.database import AuctionDatabase
from src.report_generator import ReportGenerator

def main():
    print("=" * 80)
    print("PURPLE WAVE WEEKLY TREND ANALYSIS TEST")
    print("=" * 80)
    print()
    
    # Configure DSPy to use local Ollama
    print("Configuring DSPy with Ollama (Llama 3.1 8B)...")
    lm = dspy.LM('ollama_chat/llama3.1:8b', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm)
    print("✓ DSPy configured\n")
    
    # Initialize
    print("Connecting to database...")
    db = AuctionDatabase()
    analyzer = AuctionAnalyzer(db)
    report_gen = ReportGenerator(analyzer)
    
    print("✓ Connected\n")
    
    # Test 1: Simple trend analysis
    print("\n" + "=" * 80)
    print("TEST 1: Analyze Weekly Lot Value Trends")
    print("=" * 80)
    result = analyzer.analyze_weekly_lot_value_trends(2026)
    print(result)
    
    # Test 2: Find anomalies
    print("\n" + "=" * 80)
    print("TEST 2: Find Weekly Anomalies")
    print("=" * 80)
    result = analyzer.find_weekly_anomalies(2026)
    print(result)
    
    # Test 3: Generate full report
    print("\n" + "=" * 80)
    print("TEST 3: Generate Comprehensive Report")
    print("=" * 80)
    report = report_gen.generate_comprehensive_report(2026)
    
    # Save reports
    print("\nSaving reports...")
    report_gen.save_report(report)
    report_gen.save_report_as_text(report)
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS COMPLETE")
    print("=" * 80)
    
    # Cleanup
    db.close()

if __name__ == "__main__":
    main()