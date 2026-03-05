import pandas as pd
from trino.dbapi import connect

from src.config import TRINO_HOST, TRINO_PORT, TRINO_USER, TRINO_CATALOG, TRINO_SCHEMA


def get_connection():
    return connect(
        host=TRINO_HOST,
        port=TRINO_PORT,
        user=TRINO_USER,
        catalog=TRINO_CATALOG,
        schema=TRINO_SCHEMA,
    )


def run_query(query: str, params: tuple = None) -> pd.DataFrame:
    """Run a SQL query against Trino and return a pandas DataFrame."""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params or [])
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        print(f"Database error: {str(e)}")
        return pd.DataFrame()
