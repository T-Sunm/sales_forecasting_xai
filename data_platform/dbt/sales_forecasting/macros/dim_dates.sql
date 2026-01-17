{{ config(materialized='table') }}

{{ dbt_utils.date_spine(
    datepart="day",
    start_date="cast('2012-01-01' as date)",
    end_date="cast('2014-12-31' as date)"
) }}