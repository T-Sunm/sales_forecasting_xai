{{ config(materialized='table') }}

with aggregates_data as (
    select * from {{ ref('int_store_item_aggregates') }}
),

holidays as (
    select date from {{ source('raw', 'stg_holidays') }}
),

blackfridays as (
    select date from {{ source('raw', 'stg_blackfriday') }}
),

date_features as (
    select
        d.*,
        
        -- Basic date components
        extract(year from d.date) as year,
        extract(month from d.date) as month,
        extract(day from d.date) as day,
        extract(dow from d.date) as day_of_week,  -- 0=Sunday, 6=Saturday
        extract(quarter from d.date) as quarter,
        
        -- US Meteorological Seasons
        case 
            when extract(month from d.date) in (12, 1, 2) then 0  -- Winter
            when extract(month from d.date) in (3, 4, 5) then 1   -- Spring
            when extract(month from d.date) in (6, 7, 8) then 2   -- Summer
            else 3  -- Fall (9, 10, 11)
        end as season,
        
        -- Weekend flag (Saturday=6, Sunday=0)
        case 
            when extract(dow from d.date) in (0, 6) then 1 
            else 0 
        end as is_weekend,
        
        -- Holiday flag
        case 
            when h.date is not null then 1 
            else 0 
        end as is_holiday,
        
        -- Black Friday flag
        case 
            when b.date is not null then 1 
            else 0 
        end as is_blackfriday,
        
        -- Season one-hot encoding (Fall is reference category)
        case when extract(month from d.date) in (3, 4, 5) then 1 else 0 end as season_spring,
        case when extract(month from d.date) in (6, 7, 8) then 1 else 0 end as season_summer,
        case when extract(month from d.date) in (12, 1, 2) then 1 else 0 end as season_winter,
        case when extract(month from d.date) in (9, 10, 11) then 1 else 0 end as season_fall
        
    from aggregates_data d
    left join holidays h on d.date = h.date
    left join blackfridays b on d.date = b.date
)

select * from date_features
