{% macro calculate_ewma(column_name, alpha, window_size) %}
    (
        {%- for i in range(1, window_size + 1) -%}
            {# Tính trọng số chính xác: alpha * (1-alpha)^(i-1) #}
            {%- set weight = alpha * ((1 - alpha) ** (i - 1)) -%}
            
            {# Chỉ generate SQL nếu trọng số đủ lớn (> 0.0001) để tránh query dài vô nghĩa #}
            {%- if weight > 0.0001 -%}
                COALESCE(LAG({{ column_name }}, {{ i }}) OVER w, 0) * {{ weight }}
                {%- if not loop.last and (alpha * ((1 - alpha) ** i)) > 0.0001 -%} + {%- endif -%}
            {%- endif -%}
        {%- endfor -%}
    ) 
    {# Chia cho tổng trọng số thực tế của chuỗi hữu hạn để chuẩn hóa #}
    / {{ 1 - (1 - alpha) ** window_size }}
{% endmacro %}
