with source as (
    select * from {{ source('raw', 'raw_weather') }}

),

cleaned_numeric as (
    select
        station_nbr as station_id,
        cast(date as date) as date,
        
        {{ clean_weather_numeric('tmax') }} as tmax,
        {{ clean_weather_numeric('tmin') }} as tmin,
        {{ clean_weather_numeric('tavg') }} as tavg,
        {{ clean_weather_numeric('depart') }} as depart,
        {{ clean_weather_numeric('dewpoint') }} as dewpoint,
        {{ clean_weather_numeric('wetbulb') }} as wetbulb,
        {{ clean_weather_numeric('heat') }} as heat,
        {{ clean_weather_numeric('cool') }} as cool,
        {{ clean_weather_numeric('sunrise') }} as sunrise,
        {{ clean_weather_numeric('sunset') }} as sunset,
        codesum,
        {{ clean_weather_numeric('snowfall') }} as snowfall,
        {{ clean_weather_numeric('preciptotal') }} as precip_total,
        {{ clean_weather_numeric('stnpressure') }} as stn_pressure,
        {{ clean_weather_numeric('sealevel') }} as sea_level,
        {{ clean_weather_numeric('resultspeed') }} as result_speed,
        {{ clean_weather_numeric('resultdir') }} as result_dir,
        {{ clean_weather_numeric('avgspeed') }} as avg_speed

    from source

)

select * from cleaned_numeric