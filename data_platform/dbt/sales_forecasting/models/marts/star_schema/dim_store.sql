{{ config(
    materialized='table',
    post_hook="ALTER TABLE {{ this }} ADD PRIMARY KEY (store_key)"
) }}

SELECT
    {{ dbt_utils.generate_surrogate_key(['store_id']) }} AS store_key,
    store_id,
    station_id,
    'Store ' || store_id AS store_description,
    CURRENT_TIMESTAMP AS created_at
FROM {{ ref('stg_key') }}
