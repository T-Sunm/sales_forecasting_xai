{{ config(materialized='table') }}

with base as (
  select distinct
    coalesce(is_ra, 0) as is_ra,
    coalesce(is_sn, 0) as is_sn,
    coalesce(is_fg, 0) as is_fg,
    coalesce(is_br, 0) as is_br,
    coalesce(is_ts, 0) as is_ts,
    coalesce(is_hz, 0) as is_hz,
    coalesce(is_dz, 0) as is_dz,
    coalesce(is_sq, 0) as is_sq,
    coalesce(is_fz, 0) as is_fz,
    coalesce(is_vc, 0) as is_vc
  from {{ ref('int_weather_features') }}
)

select
  md5(concat(
    cast(is_ra as varchar),
    cast(is_sn as varchar),
    cast(is_fg as varchar),
    cast(is_br as varchar),
    cast(is_ts as varchar),
    cast(is_hz as varchar),
    cast(is_dz as varchar),
    cast(is_sq as varchar),
    cast(is_fz as varchar),
    cast(is_vc as varchar)
  )) as weather_profile_key,
  *
from base
