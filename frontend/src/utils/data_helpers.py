"""
Data Helper Functions
Utility functions for data processing in frontend
"""

import pandas as pd
from typing import Optional, Tuple


def get_top_data_pair(df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the (store_nbr, item_nbr) pair with the most records
    
    Args:
        df: DataFrame containing store_nbr and item_nbr columns
        
    Returns:
        Tuple of (store_id, item_id) or (None, None) if not found
    """
    if df.empty or "store_nbr" not in df.columns or "item_nbr" not in df.columns:
        return None, None
        
    counts = (
        df.groupby(["store_nbr", "item_nbr"])
        .size()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )
    
    if counts.empty:
        return None, None
        
    top_row = counts.iloc[0]
    return int(top_row["store_nbr"]), int(top_row["item_nbr"])
