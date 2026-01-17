{{ config(materialized='table') }}

with sales_features as (
    select * from {{ ref('int_date_features') }}
),

-- Filter out cold start rows
-- Drop rows where ANY critical time series feature is NULL
clean_data as (
    select *
    from sales_features
    where 
        -- Largest lag features (if these are not null, smaller ones won't be either)
        logunits_lag_28 is not null
        
        -- Largest window rolling features (need most historical data)
        and logunits_std_28d is not null  -- stddev might be null even when mean exists
        
        -- EWMA features
        and logunits_ewma_7d_a05 is not null
        and logunits_ewma_7d_a075 is not null
        
        -- Store/Item aggregates
        and store_sum_7d is not null
        and store_mean_7d is not null
        and item_sum_7d is not null
        and item_mean_7d is not null
)

select * from clean_data
