{{ config(schema='default') }}

{%- set numeric_cols = [
  'tmax', 'tmin', 'tavg', 'depart', 'dewpoint', 'wetbulb', 'heat', 'cool',
  'sunrise', 'sunset', 'snowfall', 'preciptotal', 'stnpressure', 'sealevel',
  'resultspeed', 'resultdir', 'avgspeed'
] -%}

{%- set weather_codes = [
  'RA', 'SN', 'FG', 'BR', 'UP', 'TS', 'HZ',
  'DZ', 'SQ', 'FZ', 'MI', 'PR', 'BC', 'BL', 'VC'
] -%}

with station_means as (
  select
    station_id,
    {%- for col in numeric_cols %}
    avg({{ col }}) as {{ col }}_stn_mean{{ ',' if not loop.last }}
    {%- endfor %}
  from {{ ref('stg_weather') }}
  group by station_id
),

global_means as (
  select
    {%- for col in numeric_cols %}
    avg({{ col }}) as {{ col }}_glv_mean{{ ',' if not loop.last }}
    {%- endfor %}
  from {{ ref('stg_weather') }}
),

imputed as (
  select
    w.station_id,
    w.date,
    w.codesum,
    {%- for col in numeric_cols %}
    coalesce(w.{{ col }}, sm.{{ col }}_stn_mean, gm.{{ col }}_glv_mean) as {{ col }}{{ ',' if not loop.last }}
    {%- endfor %}
  from {{ ref('stg_weather') }} w
  left join station_means sm on w.station_id = sm.station_id
  cross join global_means gm
)

select
  station_id,
  date,
  {%- for col in numeric_cols %}
  {{ col }},
  {%- endfor %}
  {%- for code in weather_codes %}
  case when codesum like '%{{ code }}%' then 1 else 0 end as is_{{ code | lower }}{{ ',' if not loop.last }}
  {%- endfor %}
from imputed
