-- Query to check NULL counts for ALL columns in int_date_features
-- Run this in pgAdmin to see which columns have NULLs

WITH null_counts AS (
    SELECT
        COUNT(*) as total_rows,
        
        -- Sales features
        COUNT(*) - COUNT(units) as null_units,
        COUNT(*) - COUNT(log_units) as null_log_units,
        COUNT(*) - COUNT(logunits) as null_logunits,
        
        -- Lag features
        COUNT(*) - COUNT(logunits_lag_1) as null_lag_1,
        COUNT(*) - COUNT(logunits_lag_7) as null_lag_7,
        COUNT(*) - COUNT(logunits_lag_28) as null_lag_28,
        
        -- Rolling features
        COUNT(*) - COUNT(logunits_mean_7d) as null_mean_7d,
        COUNT(*) - COUNT(logunits_mean_14d) as null_mean_14d,
        COUNT(*) - COUNT(logunits_mean_28d) as null_mean_28d,
        COUNT(*) - COUNT(logunits_std_7d) as null_std_7d,
        COUNT(*) - COUNT(logunits_std_14d) as null_std_14d,
        COUNT(*) - COUNT(logunits_std_28d) as null_std_28d,
        
        -- EWMA features
        COUNT(*) - COUNT(logunits_ewma_7d_a05) as null_ewma_7d_a05,
        COUNT(*) - COUNT(logunits_ewma_7d_a075) as null_ewma_7d_a075,
        
        -- Store/Item aggregates
        COUNT(*) - COUNT(store_sum_7d) as null_store_sum_7d,
        COUNT(*) - COUNT(store_mean_7d) as null_store_mean_7d,
        COUNT(*) - COUNT(item_sum_7d) as null_item_sum_7d,
        COUNT(*) - COUNT(item_mean_7d) as null_item_mean_7d,
        
        -- Date features (should be 0)
        COUNT(*) - COUNT(year) as null_year,
        COUNT(*) - COUNT(month) as null_month,
        COUNT(*) - COUNT(is_weekend) as null_is_weekend,
        COUNT(*) - COUNT(is_holiday) as null_is_holiday,
        COUNT(*) - COUNT(is_blackfriday) as null_is_blackfriday
        
    FROM public.int_date_features
)

SELECT * FROM null_counts;
