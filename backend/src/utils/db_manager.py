from sqlalchemy import create_engine, text
import pandas as pd
from src.config import DB_URL

# Create SQLAlchemy engine
engine = create_engine(DB_URL)

def run_query(query: str, params: dict = None) -> pd.DataFrame:
    """
    Run a SQL query and return a pandas DataFrame
    """
    try:
        if params:
            # SQLAlchemy text() with named parameters
            return pd.read_sql(text(query), engine, params=params)
        else:
            return pd.read_sql(text(query), engine)
    except Exception as e:
        print(f"Database error: {str(e)}")
        # Return empty df on error to avoid crashing the API
        return pd.DataFrame()
