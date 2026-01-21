{{ config(materialized='table') }}

with cleaned_numeric as (
    select * from {{ ref('stg_weather') }}
),

imputed_data as (
    select
        station_id,
        date,
        codesum,
        coalesce(tmax, avg(tmax) over (partition by station_id), avg(tmax) over ()) as tmax,
        coalesce(tmin, avg(tmin) over (partition by station_id), avg(tmin) over ()) as tmin,
        coalesce(tavg, avg(tavg) over (partition by station_id), avg(tavg) over ()) as tavg,
        coalesce(depart, avg(depart) over (partition by station_id), avg(depart) over ()) as depart,
        coalesce(dewpoint, avg(dewpoint) over (partition by station_id), avg(dewpoint) over ()) as dewpoint,
        coalesce(wetbulb, avg(wetbulb) over (partition by station_id), avg(wetbulb) over ()) as wetbulb,
        coalesce(heat, avg(heat) over (partition by station_id), avg(heat) over ()) as heat,
        coalesce(cool, avg(cool) over (partition by station_id), avg(cool) over ()) as cool,
        coalesce(sunrise, avg(sunrise) over (partition by station_id), avg(sunrise) over ()) as sunrise,
        coalesce(sunset, avg(sunset) over (partition by station_id), avg(sunset) over ()) as sunset,
        coalesce(snowfall, avg(snowfall) over (partition by station_id), avg(snowfall) over ()) as snowfall,
        coalesce(precip_total, avg(precip_total) over (partition by station_id), avg(precip_total) over ()) as precip_total,
        coalesce(stn_pressure, avg(stn_pressure) over (partition by station_id), avg(stn_pressure) over ()) as stn_pressure,
        coalesce(sea_level, avg(sea_level) over (partition by station_id), avg(sea_level) over ()) as sea_level,
        coalesce(result_speed, avg(result_speed) over (partition by station_id), avg(result_speed) over ()) as result_speed,
        coalesce(result_dir, avg(result_dir) over (partition by station_id), avg(result_dir) over ()) as result_dir,
        coalesce(avg_speed, avg(avg_speed) over (partition by station_id), avg(avg_speed) over ()) as avg_speed
    from cleaned_numeric
),

weather_with_codes as (
    select
        *,
        {{ parse_weather_codes('codesum') }}
    from imputed_data
)

select * from weather_with_codes
