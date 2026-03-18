with source as (
    select * from {{ source('raw', 'stg_sales') }}
),

with_log as (
    select
        cast(date as date)     as date,
        store_id,
        item_id,
        cast(units as integer) as units,
        ln(cast(units as float) + 1) as log_units
    from source
)

select * from with_log
