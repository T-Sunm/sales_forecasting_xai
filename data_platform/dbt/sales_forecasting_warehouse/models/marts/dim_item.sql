{{ config(
  materialized='table',
  post_hook="alter table {{ this }} add primary key (item_id)"
) }}

select distinct
  item_id
from {{ source('intermediate', 'int_active_sales') }}
