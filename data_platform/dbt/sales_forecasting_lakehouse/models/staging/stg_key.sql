{{ config(schema='default') }}

select
  store_nbr   as store_id,
  station_nbr as station_id
from default.raw_key
