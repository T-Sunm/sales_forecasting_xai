{{ config(
  materialized='table',
  post_hook=[
    "alter table {{ this }} add foreign key (date) references {{ ref('dim_date') }}(date)",
    "alter table {{ this }} add foreign key (store_id) references {{ ref('dim_store') }}(store_id)",
    "alter table {{ this }} add foreign key (weather_profile_key) references {{ ref('dim_weather_profile') }}(weather_profile_key)"
  ]
) }}

with weather_base as (
  select distinct
    date,
    station_id,
    tmax, tmin, tavg,
    dewpoint, wetbulb,
    preciptotal, snowfall,
    resultspeed, resultdir, avgspeed,
    coalesce(is_ra, 0) as is_ra,
    coalesce(is_sn, 0) as is_sn,
    coalesce(is_fg, 0) as is_fg,
    coalesce(is_br, 0) as is_br,
    coalesce(is_up, 0) as is_up,
    coalesce(is_ts, 0) as is_ts,
    coalesce(is_hz, 0) as is_hz,
    coalesce(is_dz, 0) as is_dz,
    coalesce(is_sq, 0) as is_sq,
    coalesce(is_fz, 0) as is_fz,
    coalesce(is_mi, 0) as is_mi,
    coalesce(is_pr, 0) as is_pr,
    coalesce(is_bc, 0) as is_bc,
    coalesce(is_bl, 0) as is_bl,
    coalesce(is_vc, 0) as is_vc
  from {{ source('mart', 'sales_forecast') }}
),

with_profile as (
  select
    w.*,
    p.weather_profile_key
  from weather_base w
  left join {{ ref('dim_weather_profile') }} p
    on  w.is_ra = p.is_ra
    and w.is_sn = p.is_sn
    and w.is_fg = p.is_fg
    and w.is_br = p.is_br
    and w.is_up = p.is_up
    and w.is_ts = p.is_ts
    and w.is_hz = p.is_hz
    and w.is_dz = p.is_dz
    and w.is_sq = p.is_sq
    and w.is_fz = p.is_fz
    and w.is_mi = p.is_mi
    and w.is_pr = p.is_pr
    and w.is_bc = p.is_bc
    and w.is_bl = p.is_bl
    and w.is_vc = p.is_vc
),

final as (
  select
    w.date,
    s.store_id,
    w.station_id,
    w.weather_profile_key,
    w.tmax, w.tmin, w.tavg,
    w.dewpoint, w.wetbulb,
    w.preciptotal, w.snowfall,
    w.resultspeed, w.resultdir, w.avgspeed
  from with_profile w
  join {{ ref('dim_store') }} s
    on w.station_id = s.station_id
)

select * from final
