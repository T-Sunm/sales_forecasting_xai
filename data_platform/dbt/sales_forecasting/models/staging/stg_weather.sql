with source as (
    select * from {{ source('raw', 'stg_weather') }}

),

cleaned_numeric as (
    select
        station_id,
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
        {{ clean_weather_numeric('preciptotal') }} as preciptotal,
        {{ clean_weather_numeric('stnpressure') }} as stnpressure,
        {{ clean_weather_numeric('sealevel') }} as sealevel,
        {{ clean_weather_numeric('resultspeed') }} as resultspeed,
        {{ clean_weather_numeric('resultdir') }} as resultdir,
        {{ clean_weather_numeric('avgspeed') }} as avgspeed

    from source

)

select * from cleaned_numeric