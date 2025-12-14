# Sales Forecasting with Explainable AI (XAI)

- **Author:** Nguyen Ha DS
- **Project Type:** Proof of Concept (PoC)
- **Tech Stack:** Python, LightGBM, SHAP, Optuna, Streamlit

## Overview

- **Sales Forecasting with Explainable AI (XAI)** is a complete end-to-end proof of concept (PoC) that leverages machine learning to forecast store-level sales with transparency and interpretability.

- The project combines time series modeling with explainability tools to provide actionable insights, making it easier for business stakeholders to understand and trust the model’s predictions.

- At its core, this project builds a sales forecasting model using **LightGBM**, optimized with **Optuna**, and explained using **SHAP (SHapley Additive exPlanations)**. It culminates in a **Streamlit web application** that allows users to explore historical sales and prediction results by store.

## Key Features

- **Data Preprocessing & Cleaning:**
  Integration of multiple data sources (sales, weather), missing value handling, outlier detection.

* **Feature Engineering:**
  Over 50 crafted features including date, lag, rolling stats, and weather-based inputs.

* **Time Series Modeling:**
  Sales forecasting using LightGBM with careful temporal train/test splitting.

* **Hyperparameter Tuning:**
  Efficient model optimization via **Optuna** for enhanced performance.

* **Explainability with SHAP:**
  Interpretable model predictions with local and global SHAP value analysis.

* **Interactive Streamlit App:**
  A web interface (`app.py`) that enables users to explore store-level forecasts and historical trends.

## Deliverables

- 5 comprehensive notebooks for data processing, feature engineering, modelling and evaluation
- Trained LightGBM model
- SHAP explainability visuals - 📄 [SHAP Analysis Summary Report](docs/shap_analysis_summary_report.md)
- Streamlit app for predictions

## Project Structure

```bash
├── app.py                          # Streamlit web app for user interaction
├── check_data/
│   ├── check_data.xlsx             # Excel file for checking prediction
│   └── prediction_results.csv      # Model prediction output
├── data/
│   ├── 2016_sales.csv              # Raw sales data for 2016
│   ├── 2017_sales.csv              # Raw sales data for 2017
│   ├── feature_engineered_data_55_features.feather
│   ├── sales_data_preprocessed.csv
│   ├── weather_data.csv
│   └── weather_preprocessed.csv
├── docs/
│   ├── project_description_poc_phase.md  # Project detail description
│   └── shap_analysis_summary_report.md   # Quick summary of SHAP results
├── environment.yml                 # Environment for most systems
├── environment_macm1.yml           # Environment for Mac M1 chip
├── requirements.txt                # Nessesary libraries
├── figures/                        # SHAP plots and EDA visuals
├── models/
│   ├── feature_stats.json
│   └── sales_forecast_model.pkl   # Trained model
├── notebooks/                     # Main work for PoC phase is based on Notebooks
│   ├── 01_preprocessing.ipynb      # Proprocessing notebook
│   ├── 02_EDA.ipynb                # EDA notebook
│   ├── 03_feature_engineering.ipynb   # Feature engineer
│   ├── 04_modelling.ipynb          # Model training (base line: Prophet and better: Light GBM)
│   └── 05_explain_model.ipynb      # Explainable AI
├── src/                            # Modular source code
│   ├── data_loader/
│   ├── data_generator/
│   ├── ui_builder/
│   ├── ui_predictor/
│   └── utils/
└── README.md
```
```bash
sales_forecasting_xai/
├── data/                    # Chứa data (nên ignore trong git, dùng DVC nếu cần)
│   ├── raw/
│   └── processed/
├── models/                  # Chứa file .pkl, .bst (LightGBM models)
├── notebooks/               # Chỉ dùng để EDA, test ý tưởng (giữ 01_preprocessing.ipynb ở đây)
├── logs/                    # Folder mount ra ngoài để lưu log file từ container
├── src/                     # SOURCE CODE CHÍNH
│   ├── __init__.py
│   ├── config.py            # Quản lý Env vars, Path (tránh hardcode path)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py        # Centralized Logging config ⭐ (Yêu cầu của bạn)
│   ├── core/                # CORE LOGIC (Quan trọng nhất)
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Logic xử lý data sạch từ notebook chuyển sang
│   │   ├── model.py         # Code load model, predict
│   │   └── xai.py           # Logic tính toán SHAP/Explainability
│   ├── api/                 # BACKEND (FastAPI)
│   │   ├── __init__.py
│   │   ├── main.py          # Entry point của FastAPI
│   │   ├── schemas.py       # Pydantic models (Input/Output data validation)
│   │   └── routers/         # Chia nhỏ các api endpoint (predict, explain...)
│   └── ui/                  # FRONTEND (Streamlit) - Refactor từ code cũ
│       ├── app.py           # Entry point Streamlit
│       └── components/      # Các widget visualization
├── tests/                   # Unit test (rất quan trọng trong MLOps)
├── .env                     # Biến môi trường
├── .gitignore
├── docker-compose.yml       # Orchestration ⭐
├── Dockerfile.api           # Image cho FastAPI ⭐
├── Dockerfile.ui            # Image cho Streamlit ⭐
└── requirements.txt         # Hoặc pyproject.toml
```
## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/nguyenhads/sales_forecasting_xai.git
   cd sales_forecasting_xai
   ```

2. **Set Up Environment**

- For general systems:

  ```bash
  conda env create -f environment.yml
  conda activate sales_forecast
  ```

- For Mac M1:

  ```bash
  conda env create -f environment_macm1.yml
  conda activate sales_forecast
  ```

  _You need to install Anaconda for this setup. If not, please use the below setup instead._

- Create a virtual environment using pure python

  ```
  python -m venv .venv

  # On macOS/Linux:
  source .venv/bin/activate

  # On Windows:
  .venv\Scripts\activate

  pip install -r requirements.txt
  ```

3. **Run the notebooks**

- After activating virtual enviroments

  ```bash
  jupyter lab
  ```

4. **Generate your all dataset**

- If you preferer generating your all dataset, you can change the range of data as well as the outlier and nan values ratio.
- In this case, modify `src/data_generator/data_generator.py `, and in below `sales_forecasting_xai` folder, run the below command

  ```bash
  python src/data_generator/data_generator.py
  ```

5. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

## How It Works

1. **Data Pipeline**
   Sales and weather data are preprocessed and merged. Features are engineered and saved for model training.

2. **Model Training**
   LightGBM is trained using time-aware train/test split. Optuna tunes the model for best performance.

3. **Explainability**
   SHAP values are calculated and visualized to explain predictions at both global and local levels.

4. **User Interface**

- `app.py` allows users to:
  - View historical sales
  - Make a predictions of future sales to properly arrange the resources

## References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)
- [Streamlit](https://streamlit.io/)

## Contact

For questions or collaboration opportunities, please reach out at:
**📧 datasciencelab.ai@gmail.com**
