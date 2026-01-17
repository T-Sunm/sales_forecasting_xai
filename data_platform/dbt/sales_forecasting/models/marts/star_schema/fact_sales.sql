{{ config(
    materialized='incremental',
    unique_key='fact_sales_key',
    post_hook=[
        "ALTER TABLE {{ this }} DROP CONSTRAINT IF EXISTS fk_fact_sales_date",
        "ALTER TABLE {{ this }} ADD CONSTRAINT fk_fact_sales_date FOREIGN KEY (date_key) REFERENCES {{ ref('dim_date') }}(date_key)",
        
        "ALTER TABLE {{ this }} DROP CONSTRAINT IF EXISTS fk_fact_sales_store",
        "ALTER TABLE {{ this }} ADD CONSTRAINT fk_fact_sales_store FOREIGN KEY (store_key) REFERENCES {{ ref('dim_store') }}(store_key)",
        
        "ALTER TABLE {{ this }} DROP CONSTRAINT IF EXISTS fk_fact_sales_product",
        "ALTER TABLE {{ this }} ADD CONSTRAINT fk_fact_sales_product FOREIGN KEY (product_key) REFERENCES {{ ref('dim_product') }}(product_key)",
        
        "ALTER TABLE {{ this }} DROP CONSTRAINT IF EXISTS fk_fact_sales_weather",
        "ALTER TABLE {{ this }} ADD CONSTRAINT fk_fact_sales_weather FOREIGN KEY (weather_key) REFERENCES {{ ref('dim_weather') }}(weather_key)"
    ]
) }}

WITH source_data AS (
    SELECT * FROM {{ ref('mart_sales_weather_features') }} 
),

final AS (
    SELECT
        -- 1. Surrogate PK for Fact Table
        {{ dbt_utils.generate_surrogate_key(['store_id', 'item_id', 'date']) }} AS fact_sales_key,

        -- 2. Foreign Keys
        {{ dbt_utils.generate_surrogate_key(['store_id']) }} AS store_key,
        {{ dbt_utils.generate_surrogate_key(['item_id']) }} AS product_key,
        TO_CHAR(date, 'YYYYMMDD')::INT AS date_key,
        {{ dbt_utils.generate_surrogate_key(['station_id']) }} AS weather_key,

        -- 3. Metrics (Measures)
        units AS units_sold,
        log_units AS log_units_sold,

        -- 4. Lag Features
        logunits_lag_1 AS units_lag_1,
        logunits_lag_7 AS units_lag_7,
        logunits_lag_28 AS units_lag_28,
        
        -- 5. Rolling Features (7d/14d/28d)
        logunits_mean_7d AS units_mean_7d,
        logunits_std_7d AS units_std_7d,
        logunits_mean_14d AS units_mean_14d,
        logunits_std_14d AS units_std_14d,
        logunits_mean_28d AS units_mean_28d,
        logunits_std_28d AS units_std_28d,

        -- 6. EWMA Features
        logunits_ewma_7d_a05 AS units_ewma_7d_a05,
        logunits_ewma_7d_a075 AS units_ewma_7d_a075,
        
        -- 7. Context Aggregates
        store_sum_7d,
        store_mean_7d,
        item_sum_7d,
        item_mean_7d,
        
        -- 8. Weather Features (denormalized for fast queries)
        tmax,
        tmin,
        tavg,
        precip_total,
        is_ra,
        is_sn,
        is_fg,

        -- Metadata
        CURRENT_TIMESTAMP AS loaded_at
    FROM source_data
)

SELECT * FROM final

{% if is_incremental() %}
  WHERE date_key > (SELECT MAX(date_key) FROM {{ this }})
{% endif %}
