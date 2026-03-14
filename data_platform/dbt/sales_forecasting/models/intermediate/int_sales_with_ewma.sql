{{ config(materialized='table') }}

with lag_rolling_data as (
    select * 
    from {{ ref('int_sales_with_rolling') }}
    order by store_id, item_id, date
),

ewma_features as (
    select
        *,
        
        -- EWMA với alpha = 0.5, window = 7 days
        -- Chỉ dùng 7 ngày vì sau đó weights quá nhỏ (< 0.001)
        {{ calculate_ewma('logunits', 0.5, 7) }} as ewma7_a05,
        
        -- EWMA với alpha = 0.75, window = 7 days
        -- Alpha cao hơn = nhấn mạnh giá trị gần đây hơn
        {{ calculate_ewma('logunits', 0.75, 7) }} as ewma7_a075
        
    from lag_rolling_data
    window w as (partition by store_id, item_id order by date)
)

select * from ewma_features
