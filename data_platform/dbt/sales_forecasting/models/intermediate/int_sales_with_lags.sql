{{ config(materialized='table') }}

with base as (
    select * 
    from {{ ref('int_active_sales') }}
    order by store_id, item_id, date
),

lag_features as (
    select
        *,
        log_units as logunits,
        
        -- Lag features [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
        lag(log_units, 1) over (partition by store_id, item_id order by date) as logunits_lag_1,
        lag(log_units, 2) over (partition by store_id, item_id order by date) as logunits_lag_2,
        lag(log_units, 3) over (partition by store_id, item_id order by date) as logunits_lag_3,
        lag(log_units, 4) over (partition by store_id, item_id order by date) as logunits_lag_4,
        lag(log_units, 5) over (partition by store_id, item_id order by date) as logunits_lag_5,
        lag(log_units, 6) over (partition by store_id, item_id order by date) as logunits_lag_6,
        lag(log_units, 7) over (partition by store_id, item_id order by date) as logunits_lag_7,
        lag(log_units, 14) over (partition by store_id, item_id order by date) as logunits_lag_14,
        lag(log_units, 21) over (partition by store_id, item_id order by date) as logunits_lag_21,
        lag(log_units, 28) over (partition by store_id, item_id order by date) as logunits_lag_28
        
    from base
)

select * from lag_features