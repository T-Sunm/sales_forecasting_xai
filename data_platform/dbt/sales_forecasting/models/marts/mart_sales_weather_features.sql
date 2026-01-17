{{ config(materialized='table') }}

with sales_features as (
    select * from {{ ref('int_date_features') }}
),

store_station_mapping as (
    select 
        store_id,
        station_id
    from {{ ref('stg_key') }}
),

weather_features as (
    select * from {{ ref('int_weather_imputed') }}
),

-- Join sales with weather via store-station mapping
sales_with_weather as (
    select
        s.*,
        ss.station_id,
        -- Weather numeric features
        w.tmax,
        w.tmin,
        w.tavg,
        w.depart,
        w.dewpoint,
        w.wetbulb,
        w.heat,
        w.cool,
        w.sunrise,
        w.sunset,
        w.snowfall,
        w.precip_total,
        w.stn_pressure,
        w.sea_level,
        w.result_speed,
        w.result_dir,
        w.avg_speed,
        -- Weather code features (binary)
        w.is_ra,
        w.is_sn,
        w.is_fg,
        w.is_br,
        w.is_up,
        w.is_ts,
        w.is_hz,
        w.is_dz,
        w.is_sq,
        w.is_fz,
        w.is_mi,
        w.is_pr,
        w.is_bc,
        w.is_bl,
        w.is_vc
    from sales_features s
    left join store_station_mapping ss on s.store_id = ss.store_id
    left join weather_features w on ss.station_id = w.station_id and s.date = w.date
),

-- Filter cold start NULLs (same as original mart_sales_features)
clean_data as (
    select *
    from sales_with_weather
    where 
        logunits_lag_28 is not null
        and logunits_std_28d is not null
        and logunits_ewma_7d_a05 is not null
        and logunits_ewma_7d_a075 is not null
        and store_sum_7d is not null
        and store_mean_7d is not null
        and item_sum_7d is not null
        and item_mean_7d is not null
)

select * from clean_data
