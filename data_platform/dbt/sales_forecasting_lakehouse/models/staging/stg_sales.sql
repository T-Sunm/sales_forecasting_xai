{{ config(schema='default') }}

with raw as (
  select * from default.raw_train
)

select
  store_nbr        as store_id,
  item_nbr         as item_id,
  to_date(date)    as date,
  cast(units as int) as units
from raw
