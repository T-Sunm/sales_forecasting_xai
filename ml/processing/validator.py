from typing import List

# Target column
TARGET_COL = "log_units"

# Columns to drop (not features)
DROP_COLS = [
    'date', 'units', 'log_units', 'logunits',
    'is_kaggle_test', 'is_valid',
    'store_id', 'item_id', 'station_id', 
    'codesum', 'weather_profile_key'
]

def get_feature_cols(all_columns: List[str]) -> List[str]:
    """Filter all columns to get only the relevant feature columns"""
    return [c for c in all_columns if c not in DROP_COLS]

# Schema validation can be expanded here using Pandera or Pydantic in the future
