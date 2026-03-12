import dspy
import argparse
from shared.database import AuctionDatabase
from shared.config import configure_llm
from analyzers.nps.analyzer import NPSAnalyzer
from analyzers.nps.drill_down import NPSDrillDown
from analyzers.nps.diagnostic import NPSDiagnostic
from analyzers.nps.build_dashboard_data import NPSDashboardBuilder


def main():
    parser = argparse.ArgumentParser(description="NPS Analysis Pipeline")
    parser.add_argument("--drill-down", action="store_true", help="Run drill-down analysis after Layer 1")
    parser.add_argument("--diagnostic", action="store_true", help="Run Layer 2 diagnostic analysis")
    parser.add_argument("--dashboard", action="store_true", help="Export structured JSON into HTML dashboard")
    parser.add_argument("--dashboard-html", default="analyzers/nps/dashboard/nps_report.html", help="Dashboard HTML template path")
    parser.add_argument("--all", action="store_true", help="Run all layers")
    args = parser.parse_args()

    # Configure LLM (needed for DSPy layers and dashboard comment summaries)
    configure_llm(temperature=0.3)

    # Connect to database
    db = AuctionDatabase()

    try:
        # Layer 1: Descriptive (DSPy)
        if not args.dashboard or args.all:
            print("=" * 60)
            print("LAYER 1: DESCRIPTIVE ANALYSIS")
            print("=" * 60)
            analyzer = NPSAnalyzer(db)
            analysis = analyzer.describe_nps()
            print("\n" + analysis)

        # Drill-down (DSPy)
        if args.drill_down or args.all:
            print("\n" + "=" * 60)
            print("DRILL-DOWN ANALYSIS")
            print("=" * 60)
            drill = NPSDrillDown(db, output_dir="reports/nps")
            drill.drill_down_all()

        # Layer 2: Diagnostic (DSPy)
        if args.diagnostic or args.all:
            print("\n" + "=" * 60)
            print("LAYER 2: DIAGNOSTIC ANALYSIS")
            print("=" * 60)
            diag = NPSDiagnostic(db)
            diag.diagnose()

        # Dashboard JSON export (no DSPy needed)
        if args.dashboard or args.all:
            print("\n" + "=" * 60)
            print("DASHBOARD DATA EXPORT")
            print("=" * 60)
            builder = NPSDashboardBuilder(db)
            builder.export(html_path=args.dashboard_html)

    finally:
        db.close()


if __name__ == "__main__":
    main()