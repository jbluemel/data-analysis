import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional

class AuctionDatabase:
    # Allowlisted tables the agent can query
    ALLOWED_TABLES = [
        'itemv2',
        'weekly_metrics_summaryv2',
        'weekly_metrics_summary_enrichedv2',
        'weekly_metrics_by_categoryv2',
        'weekly_metrics_by_business_categoryv2',
        'weekly_metrics_by_industryv2',
        'weekly_metrics_by_familyv2',
        'weekly_metrics_by_regionv2',
        'weekly_metrics_by_districtv2',
        'weekly_metrics_by_territoryv2',
        'nps',
        'nps_enriched',
        'nps_bidding_all',
        'nps_nearest_transactions'
    ]

    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            port=5434,
            database="dbt_dev",
            user="dbt_user",
            password="dbt_password"
        )

    def list_tables(self) -> List[str]:
        """Return list of tables the agent can query"""
        return self.ALLOWED_TABLES

    def describe_table(self, table_name: str) -> List[Dict]:
        """Get column names and types for a table"""
        if table_name not in self.ALLOWED_TABLES:
            raise ValueError(f"Table '{table_name}' not in allowed list: {self.ALLOWED_TABLES}")

        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, [table_name])
            return cur.fetchall()

    def query(self, sql: str, limit: int = None) -> List[Dict]:
        """Execute a SELECT query against allowed tables"""
        # Basic safety: only SELECT queries
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed")

        # Add limit only if specified and not already present
        if limit and 'limit' not in sql.lower():
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return cur.fetchall()

    def close(self):
        """Close database connection"""
        self.conn.close()
