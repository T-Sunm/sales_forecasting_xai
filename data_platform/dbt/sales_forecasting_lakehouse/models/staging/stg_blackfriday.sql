{{ config(schema='default') }}

select to_date(date, 'yyyy-MM-dd') as date
from default.raw_blackfriday
