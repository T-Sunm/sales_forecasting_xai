{{ config(
  materialized='table',
  post_hook=[
    "alter table {{ this }} add foreign key (date) references {{ ref('dim_date') }}(date)",
    "alter table {{ this }} add foreign key (store_id) references {{ ref('dim_store') }}(store_id)",
    "alter table {{ this }} add foreign key (item_id) references {{ ref('dim_item') }}(item_id)"
  ]
) }}

select
  date,
  store_id,
  item_id,
  units,
  log_units,
  logunits_lag_1,
  logunits_lag_7,
  logunits_lag_14,
  logunits_lag_28
from {{ source('mart', 'sales_forecast') }}
