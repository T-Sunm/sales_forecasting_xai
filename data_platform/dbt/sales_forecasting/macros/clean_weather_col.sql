{% macro clean_weather_numeric(column_name) %}
    case 
        when {{ column_name }} is null then null
        when trim({{ column_name }}::text) in ('M', '-', '') then null
        when trim({{ column_name }}::text) = 'T' then 0.01
        else cast({{ column_name }} as numeric)
    end
{% endmacro %}

{% macro parse_weather_codes(column_name) %}

    {% set codes = [
        'RA', 'SN', 'FG', 'BR', 'UP', 
        'TS', 'HZ', 'DZ', 'SQ', 'FZ', 
        'MI', 'PR', 'BC', 'BL', 'VC'
    ] %}

    {% for code in codes %}
    case 
        when {{ column_name }} like '%{{ code }}%' then 1 
        else 0 
    end as is_{{ code | lower }}{{ "," if not loop.last }}
    {% endfor %}

{% endmacro %}