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
        coalesce(preciptotal, avg(preciptotal) over (partition by station_id), avg(preciptotal) over ()) as preciptotal,
        coalesce(stnpressure, avg(stnpressure) over (partition by station_id), avg(stnpressure) over ()) as stnpressure,
        coalesce(sealevel, avg(sealevel) over (partition by station_id), avg(sealevel) over ()) as sealevel,
        coalesce(resultspeed, avg(resultspeed) over (partition by station_id), avg(resultspeed) over ()) as resultspeed,
        coalesce(resultdir, avg(resultdir) over (partition by station_id), avg(resultdir) over ()) as resultdir,
        coalesce(avgspeed, avg(avgspeed) over (partition by station_id), avg(avgspeed) over ()) as avgspeed
    from cleaned_numeric
),

weather_with_codes as (
    select
        *,
        {{ parse_weather_codes('codesum') }}
    from imputed_data
)

select * from weather_with_codes
