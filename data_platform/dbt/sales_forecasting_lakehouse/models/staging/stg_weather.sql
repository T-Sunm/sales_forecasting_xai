{{ config(schema='default') }}

{%- set numeric_cols = [
  'tmax', 'tmin', 'tavg', 'depart', 'dewpoint', 'wetbulb', 'heat', 'cool',
  'sunrise', 'sunset', 'snowfall', 'preciptotal', 'stnpressure', 'sealevel',
  'resultspeed', 'resultdir', 'avgspeed'
] -%}

with raw as (
  select * from default.raw_weather
)

select
  station_nbr     as station_id,
  to_date(date)   as date,
  codesum,
  {%- for col in numeric_cols %}
  case
    when trim(cast({{ col }} as string)) is null
      or trim(cast({{ col }} as string)) = ''
      or trim(cast({{ col }} as string)) = 'M' then null
    when trim(cast({{ col }} as string)) = 'T' then cast(0.0 as double)
    else cast(
      regexp_replace(trim(cast({{ col }} as string)), '[^0-9\\.\\-]+', '') as double
    )
  end as {{ col }}{{ ',' if not loop.last }}
  {%- endfor %}
from raw
