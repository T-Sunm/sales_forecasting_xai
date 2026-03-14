"""
Data Helper Functions
"""

import pandas as pd


def get_top_data_pair(df: pd.DataFrame, store_col: str = "store_id", item_col: str = "item_id"):
    """
    Find the (store_id, item_id) pair with the most records
    
    Args:
        df: DataFrame to analyze
        store_col: Column name for store identifier
        item_col: Column name for item identifier
    
    Returns:
        Tuple of (store_id, item_id) or (None, None) if not found
    """
    if df.empty or store_col not in df.columns or item_col not in df.columns:
        return None, None
    
    counts = (
        df.groupby([store_col, item_col])
        .size()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )
    
    if counts.empty:
        return None, None
    
    top_row = counts.iloc[0]
    return int(top_row[store_col]), int(top_row[item_col])
