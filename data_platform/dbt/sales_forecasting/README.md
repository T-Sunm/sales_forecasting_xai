# sales_forecasting — dbt Project

This repository serves as the primary dbt analytics engineering project, responsible for orchestrating the ELT transformations that constitute the sales forecasting machine learning pipeline.

## Architectural Structure

- **`models/`**: The core SQL-based analytical transformation logic, rigidly separated into defined boundaries:
  - `staging/`: Baseline data cleaning, structural validation, and schema standardisation.
  - `marts/`: Business-logic driven models engineered specifically for ML operations (features) and downstream visualization paradigms.
- **`seeds/`**: Static datasets formulated in CSV specification containing baseline raw records (e.g., historical sales, weather metrics, system configurations).
- **`macros/`**: Reusable Jinja/SQL operational abstractions for shared logic deduplication.
- **`tests/`**: Declarative schema constraints and logical expectations enforcing data quality adherence.

## Local Execution Flow

Verify the operational status of the target PostgreSQL database via the `infra/` stack. Once validated, initialize the pipeline sequentially using `uv`:

```powershell
# Synthesize registered packages
uv run dbt deps

# Ingest raw CSV dependencies into the warehouse
uv run dbt seed

# Materialize transformation models
uv run dbt run
```

Consult the parent `../README.md` and repository root configurations for comprehensive Python environment guidelines.
