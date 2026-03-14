"""
Spark Utilities and Helper Functions

This module provides reusable utility functions for Spark transformations.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from typing import List


def add_lag_features_generic(df: DataFrame, 
                             column: str, 
                             lags: List[int],
                             partition_cols: List[str],
                             order_col: str = "date") -> DataFrame:
    """
    Add lag features for a given column.
    
    Args:
        df: Input DataFrame
        column: Column name to create lags for
        lags: List of lag periods
        partition_cols: Columns to partition by
        order_col: Column to order by (default: "date")
    
    Returns:
        DataFrame with lag features added
    """
    window = Window.partitionBy(*partition_cols).orderBy(order_col)
    
    for lag_period in lags:
        df = df.withColumn(f"{column}_lag_{lag_period}", 
                          F.lag(column, lag_period).over(window))
    
    return df


def add_rolling_features_generic(df: DataFrame,
                                 column: str,
                                 windows: List[int],
                                 partition_cols: List[str],
                                 order_col: str = "date",
                                 stats: List[str] = ["mean", "min", "max", "std"]) -> DataFrame:
    """
    Add rolling window statistics for a given column.
    
    Args:
        df: Input DataFrame
        column: Column name to calculate statistics for
        windows: List of window sizes in rows
        partition_cols: Columns to partition by
        order_col: Column to order by (default: "date")
        stats: List of statistics to calculate (mean, min, max, std)
    
    Returns:
        DataFrame with rolling features added
    """
    w_base = Window.partitionBy(*partition_cols).orderBy(order_col)
    
    for window_size in windows:
        w = w_base.rowsBetween(-window_size + 1, 0)
        
        if "mean" in stats:
            df = df.withColumn(f"{column}_mean_{window_size}d", F.avg(column).over(w))
        if "min" in stats:
            df = df.withColumn(f"{column}_min_{window_size}d", F.min(column).over(w))
        if "max" in stats:
            df = df.withColumn(f"{column}_max_{window_size}d", F.max(column).over(w))
        if "std" in stats:
            df = df.withColumn(f"{column}_std_{window_size}d", F.stddev(column).over(w))
    
    return df


def impute_numeric_columns(df: DataFrame,
                          columns: List[str],
                          partition_col: str = None,
                          strategy: str = "hierarchical") -> DataFrame:
    """
    Impute missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to impute
        partition_col: Column to partition by for station-level mean
        strategy: Imputation strategy ("hierarchical" or "global")
    
    Returns:
        DataFrame with imputed values
    """
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == "hierarchical" and partition_col:
            w_partition = Window.partitionBy(partition_col)
            w_global = Window.partitionBy()
            
            partition_mean = F.avg(F.col(col)).over(w_partition)
            global_mean = F.avg(F.col(col)).over(w_global)
            
            df = df.withColumn(col, F.coalesce(F.col(col), partition_mean, global_mean))
        else:
            # Global mean only
            global_mean = F.avg(F.col(col)).over(Window.partitionBy())
            df = df.withColumn(col, F.coalesce(F.col(col), global_mean))
    
    return df


def one_hot_encode_column(df: DataFrame,
                         column: str,
                         values: List[str],
                         prefix: str = "is",
                         separator: str = None) -> DataFrame:
    """
    One-hot encode a column based on substring matching or exact values.
    
    Args:
        df: Input DataFrame
        column: Column name to encode
        values: List of values to create binary features for
        prefix: Prefix for new column names
        separator: If provided, split column by separator before checking
    
    Returns:
        DataFrame with one-hot encoded features
    """
    base_col = column if not separator else f"{column}_split"
    
    if separator:
        df = df.withColumn(base_col, F.split(F.trim(F.coalesce(column, F.lit(""))), separator))
    
    for value in values:
        col_name = f"{prefix}_{value.lower()}"
        
        if separator:
            # Array contains check
            df = df.withColumn(col_name, 
                             F.when(F.array_contains(base_col, value), 1).otherwise(0))
        else:
            # Substring check
            df = df.withColumn(col_name,
                             F.when(F.col(column).contains(value), 1).otherwise(0))
    
    if separator:
        df = df.drop(base_col)
    
    return df


def ensure_date_column(df: DataFrame, column: str = "date") -> DataFrame:
    """
    Ensure a column is properly formatted as a date.
    
    Args:
        df: Input DataFrame
        column: Column name to convert
    
    Returns:
        DataFrame with date column properly formatted
    """
    return df.withColumn(column, F.to_date(F.col(column)))


def filter_by_total_threshold(df: DataFrame,
                              value_col: str,
                              partition_cols: List[str],
                              threshold: float = 0) -> DataFrame:
    """
    Filter rows based on total sum within partitions.
    
    Args:
        df: Input DataFrame
        value_col: Column to sum
        partition_cols: Columns to partition by
        threshold: Minimum total value to keep partition
    
    Returns:
        Filtered DataFrame
    """
    w = Window.partitionBy(*partition_cols)
    
    return (df
            .withColumn("_total", F.sum(value_col).over(w))
            .filter(F.col("_total") > threshold)
            .drop("_total"))
