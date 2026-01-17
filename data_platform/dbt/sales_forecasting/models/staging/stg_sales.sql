with source as (
    select * from {{ source('raw', 'raw_sales') }}

),

renamed as (

    select
        cast(date as date) as date,

        store_nbr as store_id,
        item_nbr as item_id,
        cast(units as integer) as units

    from source

)

select * from renamed
