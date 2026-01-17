{{ config(materialized='table') }}

WITH ewma_data AS (
    SELECT * 
    FROM {{ ref('int_sales_with_ewma') }}
),

-- Bạn có thể gộp cả store và item aggregate vào 1 SELECT nếu DB hỗ trợ nhiều window function
final_features AS (
    SELECT
        *,
        
        -- 1. STORE CONTEXT (Sức mua của cửa hàng)
        {{ rolling_agg('SUM', 'logunits', 'store_id') }} AS store_sum_7d,
        {{ rolling_agg('AVG', 'logunits', 'store_id') }} AS store_mean_7d,
        
        -- 2. ITEM CONTEXT (Độ hot của sản phẩm)
        {{ rolling_agg('SUM', 'logunits', 'item_id') }}  AS item_sum_7d,
        {{ rolling_agg('AVG', 'logunits', 'item_id') }}  AS item_mean_7d

    FROM ewma_data
)

SELECT * FROM final_features
