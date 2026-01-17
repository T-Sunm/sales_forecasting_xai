with sales_with_totals as (
    select 
        *,
        -- Tính tổng units của cặp Store-Item trên toàn bộ lịch sử
        sum(units) over (partition by store_id, item_id) as total_lifetime_units
    from {{ ref('stg_sales') }}
)

select 
    date,
    store_id,
    item_id,
    units,
    ln(units + 1) as log_units
from sales_with_totals
where total_lifetime_units > 0
