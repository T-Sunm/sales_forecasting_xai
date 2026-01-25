{{ config(materialized='table') }}

with store_station as (
  select store_id, station_id
  from {{ ref('dim_store') }}
),

weather_station_daily as (
  select *
  from {{ source('intermediate', 'weather_features') }}
)

select
  w.date,
  ss.store_id,
  ss.station_id,

  -- numeric weather
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
  w.preciptotal,
  w.stnpressure,
  w.sealevel,
  w.resultspeed,
  w.resultdir,
  w.avgspeed,

  -- one-hot codes
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

from weather_station_daily w
join store_station ss
  on w.station_id = ss.station_id
