{% macro rolling_agg(agg_func, column_name, partition_by, order_by='date', window_size=7, shift=1) %}
    {{ agg_func }}({{ column_name }}) OVER (
        PARTITION BY {{ partition_by }}
        ORDER BY {{ order_by }}
        ROWS BETWEEN {{ window_size + shift - 1 }} PRECEDING AND {{ shift }} PRECEDING
    )
{% endmacro %}
