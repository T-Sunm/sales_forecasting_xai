{{ config(materialized='table') }}

with lag_data as (
    select * 
    from {{ ref('int_sales_with_lags') }}
    order by store_id, item_id, date
),

rolling_features as (
    select
        *,
        
        -- Rolling 7 days (mean, min, max, std)
        avg(logunits_lag_1) over (partition by store_id, item_id order by date rows between 6 preceding and current row) as logunits_mean_7d,
        min(logunits_lag_1) over (partition by store_id, item_id order by date rows between 6 preceding and current row) as logunits_min_7d,
        max(logunits_lag_1) over (partition by store_id, item_id order by date rows between 6 preceding and current row) as logunits_max_7d,
        stddev(logunits_lag_1) over (partition by store_id, item_id order by date rows between 6 preceding and current row) as logunits_std_7d,
        
        -- Rolling 14 days
        avg(logunits_lag_1) over (partition by store_id, item_id order by date rows between 13 preceding and current row) as logunits_mean_14d,
        min(logunits_lag_1) over (partition by store_id, item_id order by date rows between 13 preceding and current row) as logunits_min_14d,
        max(logunits_lag_1) over (partition by store_id, item_id order by date rows between 13 preceding and current row) as logunits_max_14d,
        stddev(logunits_lag_1) over (partition by store_id, item_id order by date rows between 13 preceding and current row) as logunits_std_14d,
        
        -- Rolling 28 days
        avg(logunits_lag_1) over (partition by store_id, item_id order by date rows between 27 preceding and current row) as logunits_mean_28d,
        min(logunits_lag_1) over (partition by store_id, item_id order by date rows between 27 preceding and current row) as logunits_min_28d,
        max(logunits_lag_1) over (partition by store_id, item_id order by date rows between 27 preceding and current row) as logunits_max_28d,
        stddev(logunits_lag_1) over (partition by store_id, item_id order by date rows between 27 preceding and current row) as logunits_std_28d
        
    from lag_data
)

select * from rolling_features