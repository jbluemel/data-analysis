import dspy
import csv
import os
import html as html_lib
from datetime import datetime
from shared.database import AuctionDatabase
from typing import Dict, List, Optional


class SummarizeCommentThemes(dspy.Signature):
    """Identify the top themes from NPS detractor comments for a specific customer segment."""

    segment_name = dspy.InputField(desc="Name and description of the customer segment being analyzed")
    comments = dspy.InputField(desc="Raw NPS comments from detractors in this segment")
    theme_summary = dspy.OutputField(desc="3-5 key themes found in the comments. For each theme: name it, explain it in one sentence, and note how many comments reflect it. End with a one-sentence overall takeaway.")


# =========================================================================
# SEGMENT DEFINITIONS
# =========================================================================

SEGMENTS = [
    {
        "name": "Power Bidder Detractors (1000+ bids)",
        "description": "Most active bidders on the platform who gave low NPS scores. These are high-value users at risk of churning.",
        "where_clause": "nps_score_label = 'detractor' AND bidding_intensity_tier = 'power_bidder (1000+)'"
    },
    {
        "name": "Buyer & Seller Detractors",
        "description": "Users who both buy and sell on Purple Wave and gave low NPS scores. Dual-role users experiencing friction.",
        "where_clause": "nps_score_label = 'detractor' AND transaction_role = 'buyer_and_seller'"
    },
    {
        "name": "FY2026 Q2 Detractors (Spike Quarter)",
        "description": "Detractors from the quarter with the highest detractor rate (18.7%). Something changed this quarter.",
        "where_clause": "nps_score_label = 'detractor' AND nps_submitted_fiscal_year = 2026 AND nps_submitted_fiscal_quarter = 2"
    },
    {
        "name": "Premium Bid Detractors ($50k+ avg bid)",
        "description": "High-value bidders averaging over $50k per bid who gave low scores. These represent the highest-dollar customers.",
        "where_clause": "nps_score_label = 'detractor' AND avg_bid_value_tier = 'premium ($50k+)'"
    },
    {
        "name": "Lapsed Bidder Detractors (bid over 1 year ago)",
        "description": "Users who haven't bid in over a year and gave low scores. Potentially lost customers.",
        "where_clause": "nps_score_label = 'detractor' AND bid_recency = 'bid_over_year_ago'"
    },
    {
        "name": "Veteran Account Detractors (5+ years)",
        "description": "Long-tenured accounts who gave low scores. These users know the platform well and their feedback reflects deep experience.",
        "where_clause": "nps_score_label = 'detractor' AND account_tenure_tier = 'veteran (5+ years)'"
    },
]

ACCOUNT_COLUMNS = [
    'entity_id', 'nps_score', 'nps_score_label', 'nps_comment', 'nps_source',
    'nps_submitted_date', 'nps_submitted_fiscal_year', 'nps_submitted_fiscal_quarter',
    'nps_campaign_name', 'account_type', 'company_name', 'bidding_status',
    'copart_user', 'transaction_role', 'state', 'region_name', 'territory_name',
    'bought_item_count', 'sold_item_count', 'bought_hammer_total', 'sold_hammer_total',
    'recent_bought_date', 'recent_sold_date',
    'items_bid_on', 'lifetime_bids', 'wins', 'win_rate_pct', 'avg_highest_bid',
    'serious_bids', 'bidding_intensity_tier', 'win_rate_tier', 'avg_bid_value_tier',
    'bid_recency', 'account_tenure_tier', 'customer_lifecycle_stage', 'engagement_level'
]


class NPSDrillDown(dspy.Module):
    """Drill into each NPS insight with account-level detail and comment analysis."""

    def __init__(self, db: AuctionDatabase, output_dir: str = "reports"):
        super().__init__()
        self.db = db
        self.output_dir = output_dir
        self.summarize = dspy.ChainOfThought(SummarizeCommentThemes)
        os.makedirs(output_dir, exist_ok=True)

    def query(self, sql: str) -> List[Dict]:
        return self.db.query(sql)

    def _get_segment_accounts(self, where_clause: str) -> List[Dict]:
        """Pull all accounts for a segment with their metrics and comments."""
        cols = ", ".join(ACCOUNT_COLUMNS)
        return self.query(f"""
            SELECT {cols}
            FROM nps_enriched
            WHERE {where_clause}
            ORDER BY nps_score ASC, lifetime_bids DESC
        """)

    def _get_nearest_transactions(self, entity_ids: List[int]) -> Dict[int, List[Dict]]:
        """Pull nearest transactions for a list of entity IDs."""
        if not entity_ids:
            return {}

        id_list = ",".join(str(eid) for eid in entity_ids)
        results = self.query(f"""
            SELECT entity_id, txn_role, txn_rank, item_id, contract_price,
                   make, model, industry, category, auction_date,
                   days_from_nps, txn_proximity_label
            FROM nps_nearest_transactions
            WHERE entity_id IN ({id_list})
              AND txn_rank = 1
            ORDER BY entity_id, txn_role
        """)

        # Group by entity_id
        by_entity = {}
        for r in results:
            eid = r['entity_id']
            if eid not in by_entity:
                by_entity[eid] = []
            by_entity[eid].append(r)
        return by_entity

    def _get_segment_comments(self, where_clause: str) -> List[str]:
        """Pull just the non-empty comments for a segment."""
        results = self.query(f"""
            SELECT nps_comment
            FROM nps_enriched
            WHERE {where_clause}
              AND nps_comment IS NOT NULL
              AND trim(nps_comment) != ''
            ORDER BY nps_score ASC
        """)
        return [r['nps_comment'] for r in results]

    def _summarize_comments(self, segment_name: str, description: str, comments: List[str]) -> str:
        """Have LLM summarize comment themes."""
        if not comments:
            return "No comments available for this segment."

        comments_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))

        result = self.summarize(
            segment_name=f"{segment_name}\n{description}",
            comments=comments_text
        )
        return result.theme_summary

    # =========================================================================
    # FORMAT HELPERS
    # =========================================================================

    def _fmt_date(self, val) -> str:
        if val is None:
            return "—"
        s = str(val)[:10]
        if s == "" or s == "None":
            return "—"
        return s

    def _fmt_money(self, val) -> str:
        if val is None:
            return "—"
        try:
            return f"${float(val):,.0f}"
        except (ValueError, TypeError):
            return "—"

    def _fmt_num(self, val) -> str:
        if val is None:
            return "—"
        try:
            return f"{int(val):,}"
        except (ValueError, TypeError):
            return str(val)

    def _fmt_pct(self, val) -> str:
        if val is None:
            return "—"
        try:
            return f"{float(val):.1f}%"
        except (ValueError, TypeError):
            return "—"

    def _score_class(self, score) -> str:
        try:
            s = int(score)
        except (ValueError, TypeError):
            return ""
        if s <= 3:
            return "score-low"
        elif s <= 6:
            return "score-mid"
        return ""

    def _esc(self, val) -> str:
        if val is None:
            return ""
        return html_lib.escape(str(val))

    # =========================================================================
    # NEAREST TRANSACTION FORMATTER
    # =========================================================================

    def _fmt_nearest_txn(self, txns: Optional[List[Dict]]) -> str:
        """Format nearest transaction(s) for display."""
        if not txns:
            return '<span class="no-data">No transaction in data range</span>'

        parts = []
        for t in txns:
            role = t['txn_role']
            price = self._fmt_money(t['contract_price'])
            item_desc = " ".join(filter(None, [t.get('make'), t.get('model')]))
            if not item_desc.strip():
                item_desc = t.get('industry', 'Unknown')
            days = t['days_from_nps']
            label = t['txn_proximity_label']

            if days == 0:
                timing = "same day"
            elif days > 0:
                timing = f"{days}d before"
            else:
                timing = f"{abs(days)}d after"

            role_label = "Bought" if role == "buyer" else "Sold"
            parts.append(f'<div class="txn-line"><span class="txn-role txn-{role}">{role_label}</span> {self._esc(item_desc)} for {price} <span class="txn-timing">({timing})</span></div>')

        return "".join(parts)

    # =========================================================================
    # HTML GENERATION
    # =========================================================================

    def _generate_html(self, segments_data: List[Dict]) -> str:
        """Generate full HTML dashboard from segment data."""

        total_detractors = sum(s['total_detractors'] for s in segments_data)
        total_with_comments = sum(s['total_with_comments'] for s in segments_data)
        total_with_txns = sum(s['total_with_txns'] for s in segments_data)
        generated_date = datetime.now().strftime("%B %d, %Y %I:%M %p")

        # Build segment cards
        segment_html = ""
        for idx, seg in enumerate(segments_data):
            open_class = "open" if idx == 0 else ""
            seg_id = f"seg-{idx}"

            # Build account rows
            rows_html = ""
            for acct in seg['accounts']:
                eid = acct['entity_id']
                txns = seg['nearest_txns'].get(eid, [])
                txn_html = self._fmt_nearest_txn(txns)

                comment = acct.get('nps_comment') or ""
                comment_class = "comment-cell has-comment" if comment.strip() else "comment-cell"
                comment_display = self._esc(comment) if comment.strip() else '<span class="no-data">—</span>'

                score = acct.get('nps_score', '')
                score_cls = self._score_class(score)

                rows_html += f"""<tr>
                    <td>{self._esc(str(eid))}</td>
                    <td class="score-cell {score_cls}">{score}</td>
                    <td>{self._esc(acct.get('company_name') or '—')}</td>
                    <td>{self._fmt_date(acct.get('nps_submitted_date'))}</td>
                    <td>{self._esc(acct.get('state') or '—')}</td>
                    <td>{self._esc(acct.get('region_name') or '—')}</td>
                    <td>{self._esc(acct.get('transaction_role') or '—')}</td>
                    <td>{self._fmt_num(acct.get('bought_item_count'))}</td>
                    <td>{self._fmt_money(acct.get('bought_hammer_total'))}</td>
                    <td>{self._fmt_date(acct.get('recent_bought_date'))}</td>
                    <td>{self._fmt_num(acct.get('sold_item_count'))}</td>
                    <td>{self._fmt_money(acct.get('sold_hammer_total'))}</td>
                    <td>{self._fmt_date(acct.get('recent_sold_date'))}</td>
                    <td>{self._fmt_num(acct.get('lifetime_bids'))}</td>
                    <td>{self._fmt_num(acct.get('wins'))}</td>
                    <td>{self._fmt_pct(acct.get('win_rate_pct'))}</td>
                    <td>{self._fmt_money(acct.get('avg_highest_bid'))}</td>
                    <td class="txn-cell">{txn_html}</td>
                    <td class="{comment_class}">{comment_display}</td>
                </tr>"""

            # Theme summary with line breaks for HTML
            themes_html = self._esc(seg['theme_summary']).replace('\n', '<br>')

            segment_html += f"""
            <div class="segment {open_class}" id="{seg_id}">
              <div class="segment-header" onclick="toggle('{seg_id}')">
                <div class="segment-title-area">
                  <div class="segment-title">{self._esc(seg['name'])}</div>
                  <div class="segment-desc">{self._esc(seg['description'])}</div>
                </div>
                <div class="segment-badges">
                  <span class="badge detractors">{seg['total_detractors']} detractors</span>
                  <span class="badge comments">{seg['total_with_comments']} comments</span>
                  <span class="badge txns">{seg['total_with_txns']} w/ transactions</span>
                </div>
                <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
              </div>
              <div class="segment-body">
                <div class="themes-section">
                  <div class="themes-label">Comment Themes</div>
                  <div class="themes-text">{themes_html}</div>
                </div>
                <div class="controls-bar">
                  <input type="text" class="search-input" placeholder="Search accounts..." oninput="filterTable('{seg_id}', this.value)">
                  <div class="row-count"><span class="visible-count">{seg['total_detractors']}</span> of {seg['total_detractors']} accounts</div>
                </div>
                <div class="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th onclick="sortTable(this, 0)">Entity ID</th>
                        <th onclick="sortTable(this, 1)">Score</th>
                        <th onclick="sortTable(this, 2)">Company</th>
                        <th onclick="sortTable(this, 3)">NPS Date</th>
                        <th onclick="sortTable(this, 4)">State</th>
                        <th onclick="sortTable(this, 5)">Region</th>
                        <th onclick="sortTable(this, 6)">Role</th>
                        <th onclick="sortTable(this, 7)">Bought #</th>
                        <th onclick="sortTable(this, 8)">Bought $</th>
                        <th onclick="sortTable(this, 9)">Last Bought</th>
                        <th onclick="sortTable(this, 10)">Sold #</th>
                        <th onclick="sortTable(this, 11)">Sold $</th>
                        <th onclick="sortTable(this, 12)">Last Sold</th>
                        <th onclick="sortTable(this, 13)">Bids</th>
                        <th onclick="sortTable(this, 14)">Wins</th>
                        <th onclick="sortTable(this, 15)">Win %</th>
                        <th onclick="sortTable(this, 16)">Avg Bid</th>
                        <th>Nearest Transaction</th>
                        <th>Comment</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows_html}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NPS Drill-Down Report | Purple Wave</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface-hover: #22262f;
    --border: #2a2e3a;
    --text: #e1e4eb;
    --text-muted: #8b90a0;
    --accent: #7c5cfc;
    --accent-dim: rgba(124, 92, 252, 0.15);
    --red: #f05252;
    --red-dim: rgba(240, 82, 82, 0.12);
    --orange: #f59e0b;
    --orange-dim: rgba(245, 158, 11, 0.12);
    --green: #22c55e;
    --green-dim: rgba(34, 197, 94, 0.12);
    --blue: #3b82f6;
    --blue-dim: rgba(59, 130, 246, 0.12);
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }}

  .header {{
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }}

  .header h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
  }}

  .header .subtitle {{
    color: var(--text-muted);
    font-size: 0.9rem;
  }}

  .summary-bar {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
  }}

  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    min-width: 160px;
  }}

  .stat-card .label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
  }}

  .stat-card .value {{
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
  }}

  .stat-card .value.red {{ color: var(--red); }}
  .stat-card .value.orange {{ color: var(--orange); }}
  .stat-card .value.accent {{ color: var(--accent); }}
  .stat-card .value.blue {{ color: var(--blue); }}

  .segment {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 1.25rem;
    overflow: hidden;
    transition: border-color 0.2s;
  }}

  .segment:hover {{ border-color: var(--accent); }}

  .segment-header {{
    padding: 1.25rem 1.5rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    user-select: none;
  }}

  .segment-header:hover {{ background: var(--surface-hover); }}

  .segment-title-area {{ flex: 1; }}

  .segment-title {{
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
  }}

  .segment-desc {{
    font-size: 0.82rem;
    color: var(--text-muted);
  }}

  .segment-badges {{
    display: flex;
    gap: 0.75rem;
    margin-left: 1.5rem;
    flex-shrink: 0;
  }}

  .badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    font-weight: 500;
    white-space: nowrap;
  }}

  .badge.detractors {{ background: var(--red-dim); color: var(--red); }}
  .badge.comments {{ background: var(--accent-dim); color: var(--accent); }}
  .badge.txns {{ background: var(--blue-dim); color: var(--blue); }}

  .chevron {{
    width: 20px;
    height: 20px;
    margin-left: 1rem;
    transition: transform 0.25s;
    color: var(--text-muted);
    flex-shrink: 0;
  }}

  .segment.open .chevron {{ transform: rotate(180deg); }}

  .segment-body {{
    display: none;
    border-top: 1px solid var(--border);
  }}

  .segment.open .segment-body {{ display: block; }}

  .themes-section {{
    padding: 1.25rem 1.5rem;
    background: rgba(124, 92, 252, 0.04);
    border-bottom: 1px solid var(--border);
  }}

  .themes-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 0.6rem;
  }}

  .themes-text {{
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text);
  }}

  .controls-bar {{
    padding: 0.75rem 1.5rem;
    display: flex;
    gap: 0.75rem;
    align-items: center;
    border-bottom: 1px solid var(--border);
  }}

  .search-input {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.4rem 0.75rem;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    width: 250px;
    outline: none;
    transition: border-color 0.2s;
  }}

  .search-input:focus {{ border-color: var(--accent); }}
  .search-input::placeholder {{ color: var(--text-muted); }}

  .row-count {{
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
  }}

  .table-wrapper {{
    overflow-x: auto;
    max-height: 600px;
    overflow-y: auto;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
  }}

  thead {{ position: sticky; top: 0; z-index: 2; }}

  th {{
    background: #1e2130;
    padding: 0.6rem 0.65rem;
    text-align: left;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
    white-space: nowrap;
    border-bottom: 2px solid var(--border);
    cursor: pointer;
  }}

  th:hover {{ color: var(--accent); }}

  td {{
    padding: 0.5rem 0.65rem;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
  }}

  tr:hover td {{ background: var(--surface-hover); }}

  .score-cell {{
    font-weight: 700;
    text-align: center;
  }}

  .score-low {{ color: var(--red); }}
  .score-mid {{ color: var(--orange); }}

  .comment-cell {{
    white-space: normal;
    min-width: 250px;
    max-width: 400px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.4;
  }}

  .comment-cell.has-comment {{ color: var(--text); }}

  .txn-cell {{
    white-space: normal;
    min-width: 220px;
    max-width: 320px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
  }}

  .txn-line {{
    margin-bottom: 0.3rem;
  }}

  .txn-line:last-child {{ margin-bottom: 0; }}

  .txn-role {{
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    margin-right: 0.3rem;
  }}

  .txn-buyer {{ background: var(--green-dim); color: var(--green); }}
  .txn-seller {{ background: var(--orange-dim); color: var(--orange); }}

  .txn-timing {{
    color: var(--text-muted);
    font-size: 0.72rem;
  }}

  .no-data {{ color: #444; font-style: italic; }}
</style>
</head>
<body>

<div class="header">
  <h1>NPS Drill-Down Report</h1>
  <div class="subtitle">Purple Wave — Detractor Analysis by Segment | Generated {generated_date}</div>
</div>

<div class="summary-bar">
  <div class="stat-card">
    <div class="label">Segments Analyzed</div>
    <div class="value accent">{len(segments_data)}</div>
  </div>
  <div class="stat-card">
    <div class="label">Total Detractor Records</div>
    <div class="value red">{total_detractors:,}</div>
  </div>
  <div class="stat-card">
    <div class="label">With Comments</div>
    <div class="value orange">{total_with_comments:,}</div>
  </div>
  <div class="stat-card">
    <div class="label">With Transactions</div>
    <div class="value blue">{total_with_txns:,}</div>
  </div>
</div>

{segment_html}

<script>
function toggle(id) {{
  document.getElementById(id).classList.toggle('open');
}}

function filterTable(segId, query) {{
  const seg = document.getElementById(segId);
  const rows = seg.querySelectorAll('tbody tr');
  const q = query.toLowerCase();
  let visible = 0;
  rows.forEach(row => {{
    const text = row.textContent.toLowerCase();
    const show = text.includes(q);
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  const counter = seg.querySelector('.visible-count');
  if (counter) counter.textContent = visible;
}}

function sortTable(th, colIndex) {{
  const table = th.closest('table');
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const isAsc = th.classList.contains('sorted-asc');

  table.querySelectorAll('th').forEach(h => {{
    h.classList.remove('sorted', 'sorted-asc', 'sorted-desc');
  }});

  rows.sort((a, b) => {{
    let aVal = a.cells[colIndex].textContent.trim();
    let bVal = b.cells[colIndex].textContent.trim();
    const aNum = parseFloat(aVal.replace(/[$,%,]/g, ''));
    const bNum = parseFloat(bVal.replace(/[$,%,]/g, ''));

    if (!isNaN(aNum) && !isNaN(bNum)) {{
      return isAsc ? bNum - aNum : aNum - bNum;
    }}
    return isAsc ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
  }});

  th.classList.add('sorted', isAsc ? 'sorted-desc' : 'sorted-asc');
  rows.forEach(row => tbody.appendChild(row));
}}
</script>

</body>
</html>"""

    # =========================================================================
    # MAIN DRILL-DOWN METHOD
    # =========================================================================

    def drill_down_all(self) -> str:
        """Run drill-down analysis on all defined segments."""

        print("Running NPS Drill-Down Analysis...")
        print(f"Output directory: {self.output_dir}/")
        print(f"Analyzing {len(SEGMENTS)} segments...\n")

        segments_data = []
        all_csv_rows = []

        csv_columns = [
            'segment_name', 'segment_description',
            'segment_total_detractors', 'segment_detractors_with_comments',
            'segment_comment_themes'
        ] + ACCOUNT_COLUMNS + [
            'nearest_txn_role', 'nearest_txn_item', 'nearest_txn_price',
            'nearest_txn_make', 'nearest_txn_model', 'nearest_txn_industry',
            'nearest_txn_date', 'nearest_txn_days_from_nps', 'nearest_txn_proximity'
        ]

        full_report = []

        for i, segment in enumerate(SEGMENTS, 1):
            name = segment["name"]
            description = segment["description"]
            where = segment["where_clause"]

            print(f"[{i}/{len(SEGMENTS)}] {name}")

            # Get accounts
            accounts = self._get_segment_accounts(where)
            print(f"  Accounts: {len(accounts)}")

            # Get comments
            comments = self._get_segment_comments(where)
            print(f"  Comments: {len(comments)}")

            # Get nearest transactions
            entity_ids = [a['entity_id'] for a in accounts]
            nearest_txns = self._get_nearest_transactions(entity_ids)
            with_txns = len([eid for eid in entity_ids if eid in nearest_txns])
            print(f"  With transactions: {with_txns}")

            # Summarize comments
            print(f"  Summarizing comments...")
            theme_summary = self._summarize_comments(name, description, comments)

            # Store segment data for HTML
            segments_data.append({
                'name': name,
                'description': description,
                'total_detractors': len(accounts),
                'total_with_comments': len(comments),
                'total_with_txns': with_txns,
                'theme_summary': theme_summary,
                'accounts': accounts,
                'nearest_txns': nearest_txns,
            })

            # Build CSV rows
            for account in accounts:
                eid = account['entity_id']
                txns = nearest_txns.get(eid, [])
                nearest = txns[0] if txns else {}

                row = {
                    'segment_name': name,
                    'segment_description': description,
                    'segment_total_detractors': len(accounts),
                    'segment_detractors_with_comments': len(comments),
                    'segment_comment_themes': theme_summary,
                    'nearest_txn_role': nearest.get('txn_role', ''),
                    'nearest_txn_item': nearest.get('item_id', ''),
                    'nearest_txn_price': nearest.get('contract_price', ''),
                    'nearest_txn_make': nearest.get('make', ''),
                    'nearest_txn_model': nearest.get('model', ''),
                    'nearest_txn_industry': nearest.get('industry', ''),
                    'nearest_txn_date': nearest.get('auction_date', ''),
                    'nearest_txn_days_from_nps': nearest.get('days_from_nps', ''),
                    'nearest_txn_proximity': nearest.get('txn_proximity_label', ''),
                }
                row.update(account)
                all_csv_rows.append(row)

            # Build console report section
            section = f"""
{'=' * 60}
SEGMENT: {name}
{'=' * 60}
Description: {description}
Total Detractors: {len(accounts)}
Detractors with Comments: {len(comments)}
With Transactions in Data: {with_txns}

COMMENT THEMES:
{theme_summary}
"""
            full_report.append(section)
            print()

        # Write combined CSV
        csv_path = f"{self.output_dir}/nps_drilldown_combined.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for row in all_csv_rows:
                writer.writerow(row)
        print(f"Combined CSV saved: {csv_path} ({len(all_csv_rows)} rows)")

        # Write HTML report
        html_path = f"{self.output_dir}/nps_drilldown_report.html"
        html_content = self._generate_html(segments_data)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report saved: {html_path}")

        # Write text report
        report_text = "\n".join(full_report)
        report_path = f"{self.output_dir}/nps_drilldown_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"NPS DRILL-DOWN ANALYSIS REPORT\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(report_text)
        print(f"Text report saved: {report_path}")

        return report_text