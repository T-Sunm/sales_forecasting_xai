{{ config(
    materialized='table',
    post_hook="ALTER TABLE {{ this }} ADD PRIMARY KEY (date_key)"
) }}

WITH date_spine AS (
  SELECT 
    '2012-01-01'::DATE + SEQUENCE.DAY AS date_day
  FROM GENERATE_SERIES(0, 1500) AS SEQUENCE(DAY)
)

SELECT
  TO_CHAR(date_day, 'YYYYMMDD')::INT AS date_key,
  date_day AS calendar_date,
  EXTRACT(YEAR FROM date_day)::INT AS year,
  EXTRACT(QUARTER FROM date_day)::INT AS quarter,
  EXTRACT(MONTH FROM date_day)::INT AS month,
  TO_CHAR(date_day, 'Month') AS month_name,
  EXTRACT(DAY FROM date_day)::INT AS day_of_month,
  EXTRACT(DOW FROM date_day)::INT AS day_of_week,
  TO_CHAR(date_day, 'Day') AS day_name,
  CASE WHEN EXTRACT(DOW FROM date_day) IN (0, 6) THEN TRUE ELSE FALSE END AS is_weekend
FROM date_spine
WHERE date_day <= '2014-12-31'::DATE
