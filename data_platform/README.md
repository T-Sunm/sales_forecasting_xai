# Data Platform (dbt + PostgreSQL)

This project manages the data transformation pipeline using **dbt** (data build tool) and **PostgreSQL**.

## 📋 Prerequisites

- Python (3.10+)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- PostgreSQL (Running locally on port 5432)

## 🚀 Setup Guide

### 1. Configure dbt Profile

Create or update your global dbt profile to connect to your local PostgreSQL instance.

**Edit Profile:**

```powershell
# Windows
notepad $env:USERPROFILE\.dbt\profiles.yml

# Linux/macOS
nano ~/.dbt/profiles.yml
```

**Configuration:**

```yaml
sales_forecasting:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      user: postgres
      password: changeme  # Update with your local password
      port: 5432
      dbname: postgres
      schema: public
      threads: 4
```

### 2. Install Dependencies

Navigate to the `data_platform` directory and sync dependencies using `uv`.

```powershell
cd data_platform
uv sync
```

### 3. Load Raw Data (ETL)

We use a Python script to load raw CSV data into the PostgreSQL `raw` schema.

- **Source Data:** `shared/data/data_raw/*.csv`
- **Destination:** PostgreSQL `raw` schema (`raw_sales`, `raw_weather`, `raw_key`)

Run the loader script:

```powershell
uv run python postgres/scripts/load_raw_data.py
```

**Expected Output:**

```
✅ Schema 'raw' created/verified
✅ Loaded train.csv -> raw.raw_sales
✅ Loaded weather.csv -> raw.raw_weather
✅ Loaded key.csv -> raw.raw_key
🎉 All data loaded successfully!
```

### 4. Verify Connection

Check if dbt can connect to your PostgreSQL database.

```powershell
cd dbt/sales_forecasting
uv run dbt debug
```

Look for: `All checks passed!`

## 🛠️ Development Workflow

### Running Models

Transform raw data into analytics-ready models:

```powershell
# Run all models
uv run dbt run

# Run specific models (e.g., only staging)
uv run dbt run --select tag:staging
```

### 5. Generate Base Models (Optional)

Instead of writing YAML files manually, use `codegen` to automatically generate the initial `sources.yml` from your database schema.

**1. Install dbt packages:**
Ensure `packages.yml` exists and includes `dbt-labs/codegen`, then run:

```powershell
```powershell
uv run dbt deps
```

**2. Create Model Directories:**
Make sure the folder structure exists:

```powershell
# PowerShell
# Note: In PowerShell, you can create multiple directories like this:
mkdir models/staging, models/intermediate, models/marts
```

**3. Generate Source YAML:**
Run this command to inspect the raw schema and output the YAML configuration:

```powershell
# Print YAML to terminal (Copy & Paste the output to models/staging/sources.yml)
uv run dbt run-operation generate_source --args '{"schema_name": "raw", "database_name": "postgres"}'
```

**Result (`models/staging/sources.yml`):**

```yaml
version: 2

sources:
  - name: raw
    database: postgres
    schema: raw
    tables:
      - name: raw_key
      - name: raw_sales
      - name: raw_weather
```

### 6. Configure Project Structure (Recommended)

Clean up the default dbt examples and configure the materialization strategies for your data layers.

**1. Remove Example Models:**
Delete the default `example` folder to keep the project clean.

```powershell
Remove-Item -Recurse -Force models/example
```

**2. Configure Materialization:**
Update `dbt_project.yml` to define how models in each layer should be built (View vs Table).

```yaml
models:
  sales_forecasting:
    # staging & intermediate -> Views (Faster, less storage)
    staging:
      +materialized: view
    intermediate:
      +materialized: view
      
    # marts -> Tables (Pre-computed for performance)
    marts:
      +materialized: table
```

## 📂 Project Structure

```text
data_platform/
├── dbt/
│   └── sales_forecasting/    # Main dbt project
│       ├── models/           # SQL transformations
│       ├── seeds/            # Static reference data
│       └── dbt_project.yml   # Project configuration
├── postgres/
│   └── scripts/
│       └── load_raw_data.py  # Script to load CSVs to Postgres
└── pyproject.toml            # Python dependencies (dbt-core, psycopg2, etc.)
```

## 📚 Resources

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt Discourse](https://discourse.getdbt.com/)
- [dbt Slack](https://community.getdbt.com/)
