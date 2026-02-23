{{ config(schema='default') }}

with base as (
  select
    *,
    sum(units) over (partition by store_id, item_id) as total_lifetime_units
  from {{ ref('stg_sales') }}
)

select
  date,
  store_id,
  item_id,
  units,
  ln(units + 1) as log_units
from base
where total_lifetime_units > 0
