from datetime import datetime
from src.analyzer import AuctionAnalyzer
from src.database import AuctionDatabase
import json

class ReportGenerator:
    def __init__(self, agent: AuctionAnalyzer):
        self.agent = agent
    
    def generate_category_report(self) -> dict:
        """Generate comprehensive category-based auction report"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "report_type": "category_auction_analysis",
            "analyses": {}
        }
        
        # Analyze all categories
        print("Generating category report...")
        categories = self.agent.db.get_all_categories()
        
        for category in categories:
            print(f"  Analyzing {category}...")
            analysis = self.agent.analyze_single_category(category)
            
            # Get stats for the report
            stats = self.agent.db.get_category_stats(category)
            
            report["analyses"][category] = {
                "analysis": analysis,
                "stats": {
                    "count": stats["count"],
                    "avg_price": float(stats["avg_price"]),
                    "total_fees": float(stats["total_fees"])
                }
            }
        
        return report
    
    def generate_weekly_trends_report(self, fiscal_year: int = 2026) -> dict:
        """Generate weekly trend analysis report"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "report_type": "weekly_trends_analysis",
            "fiscal_year": fiscal_year,
        }
        
        print(f"Generating weekly trends report for FY{fiscal_year}...")
        
        # Get trend analysis
        print("  Analyzing trends...")
        report["trend_analysis"] = self.agent.analyze_weekly_lot_value_trends(fiscal_year)
        
        # Get anomalies
        print("  Identifying anomalies...")
        report["anomalies"] = self.agent.find_weekly_anomalies(fiscal_year)
        
        # Get summary stats
        summary_stats = self.agent.db.get_weekly_stats_summary(fiscal_year)
        report["summary_stats"] = {
            "total_weeks": summary_stats["total_weeks"],
            "avg_lot_value_overall": float(summary_stats["avg_lot_value_overall"]),
            "total_revenue_fy": float(summary_stats["total_revenue_fy"]),
            "total_items_fy": summary_stats["total_items_fy"],
            "total_bids_fy": summary_stats["total_bids_fy"]
        }
        
        return report
    
    def generate_comprehensive_report(self, fiscal_year: int = 2026) -> dict:
        """Generate full executive report with trends, anomalies, and recommendations"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "report_type": "comprehensive_executive_report",
            "fiscal_year": fiscal_year,
        }
        
        print(f"Generating comprehensive report for FY{fiscal_year}...")
        
        # Get full weekly report (includes trends + anomalies + recommendations)
        print("  Generating executive summary...")
        report["executive_summary"] = self.agent.generate_full_weekly_report(fiscal_year)
        
        # Get raw data for reference
        weekly_data = self.agent.db.get_weekly_metrics(fiscal_year=fiscal_year)
        report["weekly_data"] = [
            {
                "week": w["fiscal_week_number"],
                "week_start": str(w["week_start_date"]),
                "avg_lot_value": w["avg_lot_value"],
                "total_items": w["total_items_sold"],
                "total_revenue": w["total_revenue"],
                "total_bids": w["total_bids"]
            }
            for w in weekly_data
        ]
        
        return report
    
    def save_report(self, report: dict, filepath: str = None):
        """Save report to file"""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_type = report.get("report_type", "report")
            filepath = f"reports/{report_type}_{timestamp}.json"
        
        # Create reports directory if it doesn't exist
        import os
        os.makedirs("reports", exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to: {filepath}")
        return filepath
    
    def save_report_as_text(self, report: dict, filepath: str = None):
        """Save report as readable text file"""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_type = report.get("report_type", "report")
            filepath = f"reports/{report_type}_{timestamp}.txt"
        
        # Create reports directory if it doesn't exist
        import os
        os.makedirs("reports", exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"AUCTION ANALYSIS REPORT\n")
            f.write(f"Generated: {report['generated_at']}\n")
            f.write(f"Report Type: {report['report_type']}\n")
            f.write("=" * 80 + "\n\n")
            
            if report["report_type"] == "weekly_trends_analysis":
                f.write("WEEKLY TRENDS ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(report["trend_analysis"] + "\n\n")
                
                f.write("ANOMALY DETECTION\n")
                f.write("-" * 80 + "\n")
                f.write(report["anomalies"] + "\n\n")
                
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 80 + "\n")
                stats = report["summary_stats"]
                f.write(f"Total Weeks: {stats['total_weeks']}\n")
                f.write(f"Avg Lot Value: ${stats['avg_lot_value_overall']:,.2f}\n")
                f.write(f"Total Revenue: ${stats['total_revenue_fy']:,.2f}\n")
                f.write(f"Total Items: {stats['total_items_fy']}\n")
            
            elif report["report_type"] == "comprehensive_executive_report":
                f.write(report["executive_summary"] + "\n\n")
        
        print(f"✓ Text report saved to: {filepath}")
        return filepath