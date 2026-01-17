{{ config(
    materialized='table',
    post_hook="ALTER TABLE {{ this }} ADD PRIMARY KEY (weather_key)"
) }}

SELECT
    {{ dbt_utils.generate_surrogate_key(['station_id']) }} AS weather_key,
    station_id,
    'Station ' || station_id AS station_description,
    CURRENT_TIMESTAMP AS created_at
FROM (
    SELECT DISTINCT station_id 
    FROM {{ ref('stg_weather') }}
) AS distinct_stations
