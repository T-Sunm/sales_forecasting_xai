with source as (
    select * from {{ source('raw', 'raw_key') }}
),

renamed as (
    select
        store_nbr as store_id,
        station_nbr as station_id
    from source
)

select * from renamed