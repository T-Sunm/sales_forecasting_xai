# Sales Forecasting (Legacy dbt Project)

⚠️ **This is the LEGACY implementation using pure dbt for all transformations.**

## Status: ARCHIVED

This project has been superseded by the new lakehouse architecture:
- **New project**: `dbt/sales_forecasting_warehouse/`
- **Reason**: Better separation of concerns, scalability with Spark

## What This Project Did

This was the original implementation that used **dbt only** for the entire data pipeline:

### Layers
1. **Staging** (`models/staging/`):
   - `stg_sales.sql`: Clean and rename sales data
   - `stg_weather.sql`: Clean weather data with custom macros
   - `stg_key.sql`: Store-station mapping

2. **Intermediate** (`models/intermediate/`):
   - `int_active_sales.sql`: Filter active store-item pairs
   - `int_sales_with_lags.sql`: Lag features (1, 2, 3, 7, 14, 21, 28 days)
   - `int_sales_with_rolling.sql`: Rolling window aggregations
   - `int_sales_with_ewma.sql`: Exponential weighted moving averages
   - `int_store_item_aggregates.sql`: Store/Item context features
   - `int_date_features.sql`: Date and holiday features
   - `int_weather_imputed.sql`: Weather imputation and encoding

3. **Marts** (`models/marts/`):
   - `mart_sales_features.sql`: Final feature table for ML
   - `mart_sales_weather_features.sql`: Features with weather data
   - `star_schema/`: Dimensional model for BI

### Limitations

- ❌ All transformations in PostgreSQL (not scalable)
- ❌ Complex window functions in SQL (hard to maintain)
- ❌ No separation between heavy processing and serving
- ❌ Limited to single-machine PostgreSQL capacity

## Migration to New Architecture

The logic from this project has been migrated to:

| Old Location | New Location | Technology |
|--------------|--------------|------------|
| `models/staging/` | `spark/jobs/staging/` | PySpark |
| `models/intermediate/` | `spark/jobs/intermediate/` | PySpark |
| `models/marts/` | `dbt/sales_forecasting_warehouse/models/marts/` | dbt + PostgreSQL |

## Running This Legacy Project

If you need to run this for reference:

```powershell
cd dbt/sales_forecasting

# Install dependencies
uv run dbt deps

# Run models
uv run dbt run

# Test
uv run dbt test
```

## Resources

- See `../sales_forecasting_warehouse/` for the new implementation
- See `../../README.md` for overall architecture
