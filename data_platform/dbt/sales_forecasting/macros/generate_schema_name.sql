{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name == 'public' -%}
        public
    {%- elif custom_schema_name is not none -%}
        {{ custom_schema_name }}
    {%- else -%}
        {{ default__generate_schema_name(custom_schema_name, node) }}
    {%- endif -%}
{%- endmacro %}
