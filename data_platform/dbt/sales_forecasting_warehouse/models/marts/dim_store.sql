{{ config(
  materialized='table',
  post_hook="alter table {{ this }} add primary key (store_id)"
) }}

select distinct
  store_id,
  station_id
from {{ source('raw', 'stg_key') }}
