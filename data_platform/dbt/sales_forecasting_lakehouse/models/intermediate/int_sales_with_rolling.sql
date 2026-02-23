{{ config(schema='default') }}

{%- set windows = [7, 14, 28] -%}

select
  date,
  store_id,
  item_id,
  {%- for w in windows %}
  avg(logunits_lag_1) over (
    partition by store_id, item_id order by date
    rows between {{ w - 1 }} preceding and current row
  ) as roll_avg_{{ w }}d,
  min(logunits_lag_1) over (
    partition by store_id, item_id order by date
    rows between {{ w - 1 }} preceding and current row
  ) as roll_min_{{ w }}d,
  max(logunits_lag_1) over (
    partition by store_id, item_id order by date
    rows between {{ w - 1 }} preceding and current row
  ) as roll_max_{{ w }}d,
  stddev(logunits_lag_1) over (
    partition by store_id, item_id order by date
    rows between {{ w - 1 }} preceding and current row
  ) as roll_std_{{ w }}d{{ ',' if not loop.last }}
  {%- endfor %}
from {{ ref('int_sales_with_lags') }}
