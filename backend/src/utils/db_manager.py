import logging

import pandas as pd
from trino.dbapi import connect
from trino.exceptions import DatabaseError, OperationalError

from src.config import TRINO_HOST, TRINO_PORT, TRINO_USER, TRINO_CATALOG, TRINO_SCHEMA

logger = logging.getLogger(__name__)


class TrinoServiceError(Exception):
    """Raised when a Trino query fails. Converted to HTTP 503 by the app exception handler."""
    pass


def get_connection():
    return connect(
        host=TRINO_HOST,
        port=TRINO_PORT,
        user=TRINO_USER,
        catalog=TRINO_CATALOG,
        schema=TRINO_SCHEMA,
    )


def run_query(query: str, params: tuple = None) -> pd.DataFrame:
    """Run a SQL query against Trino. Raises TrinoServiceError on DB failures."""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, list(params) if params else [])
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return pd.DataFrame(rows, columns=columns)
    except (DatabaseError, OperationalError) as e:
        logger.error("Trino query failed: %s | query=%s", e, query[:200])
        raise TrinoServiceError(str(e)) from e
