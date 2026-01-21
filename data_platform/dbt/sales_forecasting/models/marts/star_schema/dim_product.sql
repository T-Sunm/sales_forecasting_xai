{{ config(
    materialized='table',
    post_hook="ALTER TABLE {{ this }} ADD PRIMARY KEY (product_key)"
) }}

SELECT
    {{ dbt_utils.generate_surrogate_key(['item_id']) }} AS product_key,
    item_id,
    'Product ' || item_id AS product_description,
    TRUE AS is_weather_sensitive,
    CURRENT_TIMESTAMP AS created_at
FROM (SELECT DISTINCT item_id FROM {{ ref('stg_sales') }}) AS distinct_items
