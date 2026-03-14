{{ config(schema='default') }}

{%- set lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28] -%}

select
  date,
  store_id,
  item_id,
  units,
  log_units,
  {%- for k in lags %}
  lag(log_units, {{ k }}) over (partition by store_id, item_id order by date) as logunits_lag_{{ k }}{{ ',' if not loop.last }}
  {%- endfor %}
from {{ ref('int_active_sales') }}
