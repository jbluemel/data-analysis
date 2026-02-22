import dspy
import csv
from datetime import datetime
from config import configure_llm
from src.database import AuctionDatabase
from src.analyzer import AuctionAnalyzer

# Configure Claude
configure_llm(provider='claude', temperature=0.3)

# Create analyzer
db = AuctionDatabase()
agent = AuctionAnalyzer(db)

def fmt_dollar(val):
    if val is None:
        return "N/A"
    return f"${val:,.2f}"

def fmt_pct(val):
    if val is None:
        return "N/A"
    return f"{val}%"

def fmt_int(val):
    if val is None:
        return 0
    return int(val)

# Get FY26 weeks with auctions only
print("Fetching FY 2026 weeks from enriched table...")
weeks = agent.query("""
    SELECT
        fiscal_year,
        fiscal_week_number,
        total_items_sold,
        avg_lot_value,
        goal_status,
        construction_pct,
        construction_alv,
        ag_pct,
        ag_alv,
        passenger_pct,
        passenger_alv,
        trucks_med_heavy_pct,
        trucks_med_heavy_alv,
        pct_items_10k_plus,
        region_3_pct,
        region_3_alv,
        region_4_pct,
        region_4_alv,
        region_5_pct,
        region_5_alv,
        alv_target,
        alv_variance,
        alv_variance_pct,
        top_revenue_industry,
        top_revenue_pct,
        top_revenue_alv,
        anomaly_industry,
        anomaly_alv,
        anomaly_pct_above_typical
    FROM weekly_metrics_summary_enrichedv2
    WHERE fiscal_year = 'FY 2026'
      AND has_auctions = true
    ORDER BY fiscal_week_number
""")

print(f"Found {len(weeks)} weeks to analyze")

# Prepare CSV output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"reports/fy26_weekly_explanations_{timestamp}.csv"

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # ========== HEADER ==========
    writer.writerow(['PURPLE WAVE WEEKLY ALV ANALYSIS'])
    writer.writerow(['Goal: $10,000 Average Lot Value'])
    writer.writerow([])
    writer.writerow(['ML-VALIDATED DRIVERS (94.4% accuracy with geography):'])
    writer.writerow(['Priority', 'Driver', 'Threshold', 'Notes'])
    writer.writerow(['1 (58.8%)', 'Region 3 ALV', '> $8,410', 'Volume region - quality here drives success'])
    writer.writerow(['2 (13.2%)', 'Region 4 ALV', '> $9,131', 'Second volume region'])
    writer.writerow(['3', 'Construction ALV', '> $15,452', 'Top industry driver'])
    writer.writerow(['4', 'Ag ALV', '> $16,887', 'Hero when Region 3 weak'])
    writer.writerow(['5', 'Passenger %', '< 19.1%', 'Dilution risk'])
    writer.writerow(['6', 'Region 3 %', '< 70%', 'Over-concentration risk'])
    writer.writerow([])
    writer.writerow(['Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    writer.writerow([])

    # ========== DATA HEADER ==========
    columns = [
        'Fiscal Year', 'Week', 'Items Sold', 'Avg Lot Value', 'Goal Status',
        'Target', 'Variance $', 'Variance %',
        'Region 3 %', 'Region 3 ALV', 'Region 4 %', 'Region 4 ALV', 'Region 5 %', 'Region 5 ALV',
        'Construction %', 'Construction ALV',
        'Ag %', 'Ag ALV',
        'Passenger %', 'Passenger ALV',
        'Trucks Med/Heavy %', 'Trucks Med/Heavy ALV',
        'Items 10k+ %',
        'Top Revenue Industry', 'Top Revenue %', 'Top Revenue ALV',
        'Anomaly Industry', 'Anomaly ALV', 'Anomaly % Above Typical',
        'Analysis'
    ]
    writer.writerow(columns)

    # ========== DATA ROWS ==========
    for i, week in enumerate(weeks):
        fiscal_year = week['fiscal_year']
        fiscal_week = int(week['fiscal_week_number'])

        print(f"[{i+1}/{len(weeks)}] Explaining {fiscal_year} Week {fiscal_week}...")

        try:
            explanation = agent.explain_week(fiscal_year=fiscal_year, fiscal_week=fiscal_week)
        except Exception as e:
            explanation = f"ERROR: {str(e)}"
            print(f"  Error: {e}")

        row = [
            week['fiscal_year'],
            int(week['fiscal_week_number']),
            fmt_int(week['total_items_sold']),
            fmt_dollar(week['avg_lot_value']),
            week['goal_status'].upper() if week['goal_status'] else 'N/A',
            fmt_dollar(week['alv_target']),
            fmt_dollar(week['alv_variance']),
            fmt_pct(week['alv_variance_pct']),
            fmt_pct(week['region_3_pct']),
            fmt_dollar(week['region_3_alv']),
            fmt_pct(week['region_4_pct']),
            fmt_dollar(week['region_4_alv']),
            fmt_pct(week['region_5_pct']),
            fmt_dollar(week['region_5_alv']),
            fmt_pct(week['construction_pct']),
            fmt_dollar(week['construction_alv']),
            fmt_pct(week['ag_pct']),
            fmt_dollar(week['ag_alv']),
            fmt_pct(week['passenger_pct']),
            fmt_dollar(week['passenger_alv']),
            fmt_pct(week['trucks_med_heavy_pct']),
            fmt_dollar(week['trucks_med_heavy_alv']),
            fmt_pct(week['pct_items_10k_plus']),
            week['top_revenue_industry'] or 'N/A',
            fmt_pct(week['top_revenue_pct']),
            fmt_dollar(week['top_revenue_alv']),
            week['anomaly_industry'] or '',
            fmt_dollar(week['anomaly_alv']) if week['anomaly_alv'] else '',
            fmt_pct(week['anomaly_pct_above_typical']) if week['anomaly_pct_above_typical'] else '',
            explanation
        ]

        writer.writerow(row)

print(f"\n{'='*60}")
print(f"COMPLETE: {len(weeks)} weeks analyzed")
print(f"Output: {output_file}")
print(f"{'='*60}")

db.close()
