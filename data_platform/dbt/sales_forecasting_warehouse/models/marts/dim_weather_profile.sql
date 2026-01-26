{{ config(
  materialized='table',
  post_hook="alter table {{ this }} add primary key (weather_profile_key)"
) }}

with base as (
  select distinct
    -- flags
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
  from {{ source('intermediate', 'weather_features') }}
),

final as (
  select
    {{ dbt_utils.generate_surrogate_key([
      'is_ra','is_sn','is_fg','is_br','is_up','is_ts','is_hz','is_dz','is_sq','is_fz','is_mi','is_pr','is_bc','is_bl','is_vc'
    ]) }} as weather_profile_key,
    *
  from base
)

select *
from final
