# Backend API Service

FastAPI-based backend service for Sales Forecasting and XAI explanations.

## Features
- **Prediction API**: Serve model predictions (LightGBM/XGBoost).
- **XAI API**: Provide SHAP-based explanations for model decisions.
- **Health Check**: Monitor service status.

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Server
```bash
# Using Python script
python run.py

# Or via Uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
The API documentation will be available at `http://localhost:8000/docs`.
