# Spark ETL Jobs

This directory contains PySpark jobs for data transformation pipeline.

## Structure

```
spark/
├── src/                  # Reusable utilities and helper functions
├── jobs/
│   ├── staging/         # Bronze → Silver transformations
│   └── intermediate/    # Silver → Gold feature engineering
└── configs/             # Configuration files
```

## Running Jobs

### Local Development

```bash
# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/spark"

# Run staging job
spark-submit \
  --master local[*] \
  spark/jobs/staging/staging_transform.py

# Run intermediate jobs
spark-submit \
  --master local[*] \
  spark/jobs/intermediate/sales_features_pipeline.py

spark-submit \
  --master local[*] \
  spark/jobs/intermediate/weather_features_pipeline.py
```

### With Docker/Spark Cluster

```bash
# Submit to Spark cluster
spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  spark/jobs/intermediate/sales_features_pipeline.py
```

## Configuration

All paths and parameters are centralized in `configs/config.py`:
- MinIO/S3 credentials
- Bucket names and paths
- Feature engineering parameters
- Spark configurations

## Utilities

The `src/utils.py` module provides reusable functions:
- `add_lag_features_generic()` - Generic lag feature creation
- `add_rolling_features_generic()` - Rolling window statistics
- `impute_numeric_columns()` - Missing value imputation
- `one_hot_encode_column()` - One-hot encoding
- `filter_by_total_threshold()` - Partition-based filtering

## Migration from legacy project

This Spark implementation replaces the staging and intermediate layers from the legacy transformation project

| Layer | Legacy | Current Lakehouse |
|-------|-----------|-------------|
| Staging | legacy staging models | `spark/jobs/staging/` |
| Intermediate | legacy intermediate models | `spark/jobs/intermediate/` |
| Marts | legacy marts layer | `dbt/sales_forecasting_lakehouse/models/marts/` |

## Development

### Adding New Jobs

1. Create new file in appropriate folder (`staging/` or `intermediate/`)
2. Import from `configs.config` for configuration
3. Use utilities from `src/utils.py` when possible
4. Follow naming convention: `{entity}_{transformation}_pipeline.py`

### Testing

```python
# Example unit test structure
import pytest
from pyspark.sql import SparkSession
from jobs.intermediate.sales_features_pipeline import filter_active_sales

@pytest.fixture
def spark():
    return SparkSession.builder.master("local[1]").getOrCreate()

def test_filter_active_sales(spark):
    # Test implementation
    pass
```
