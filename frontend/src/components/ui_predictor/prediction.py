from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from services import get_api_client
from utils.data_helpers import get_top_data_pair
from config import COL_STORE_ID, COL_ITEM_ID


def sales_prediction_view(api_client):
    """
    Display the sales prediction tool interface
    
    Args:
        api_client: APIClient instance for backend communication
    """
    st.title("Sales Prediction Tool")

    # Sidebar: Store and Item selection via API
    store_id, item_id = create_product_selection_sidebar_api(api_client)

    if store_id is None or item_id is None:
        st.warning("⚠️ Please select both Store and Item from sidebar")
        return

    # Main form content
    st.subheader("Prediction Parameters")
    prediction_inputs = collect_prediction_inputs()

    # Make prediction button
    if st.button("Predict Sales", type="primary"):
        generate_prediction(
            api_client,
            store_id,
            item_id,
            prediction_inputs
        )


def create_product_selection_sidebar_api(api):
    """Create sidebar for store and product selection using API"""
    
    # 1. Get Top Pair for default selection
    top_res = api.get_top_pair()
    top_store = top_res.get("store_id")
    top_item = top_res.get("item_id")
    
    with st.sidebar:
        st.header("Product Selection")

        # 2. Store Selection
        stores = api.get_stores_list()
        if not stores:
            st.error("No stores available.")
            return None, None
            
        default_store_idx = 0
        if top_store in stores:
            default_store_idx = stores.index(top_store)
            
        store_id = st.selectbox("Select Store ID", options=stores, index=default_store_idx)

        # 3. Item Selection
        store_items = api.get_items_list(store_id)
        if not store_items:
            st.warning(f"No items found for Store {store_id}")
            return store_id, None
            
        default_item_idx = 0
        if top_item in store_items:
            default_item_idx = store_items.index(top_item)
            
        item_id = st.selectbox("Select Product ID", options=store_items, index=default_item_idx)

    return store_id, item_id


def generate_prediction(
    api_client,
    store_id,
    item_id,
    prediction_inputs
):
    """
    Generate sales prediction using Backend API and display results
    """
    with st.spinner("🔮 Generating prediction via backend API..."):
        try:
            # Extract date object for display (if exists)
            date_obj = prediction_inputs.pop("_date_obj", None)
            
            # Call backend API
            result = api_client.predict(
                store_id=store_id,
                item_id=item_id,
                prediction_input=prediction_inputs
            )
            
            if not result or result.get("error"):
                error_msg = result.get("error", "Unknown error") if result else "No response from backend"
                st.error(f"❌ Prediction failed: {error_msg}")
                return
            
            prediction_value = result.get("prediction_value")
            if prediction_value is None:
                st.error("❌ No prediction value returned from backend")
                return
            
            # Restore date object for display
            if date_obj:
                prediction_inputs["_date_obj"] = date_obj
            
            # Fetch historical context from API instead of local DF
            hist_res = api_client.get_historical_data(store_id=store_id, item_id=item_id, limit=100)
            historical_df = pd.DataFrame(hist_res.get("data", []))
            if not historical_df.empty:
                historical_df["date"] = pd.to_datetime(historical_df["date"])

            display_prediction_results(
                api_client,
                prediction_value,
                store_id,
                item_id,
                prediction_inputs,
                historical_df,
                forecast_history=None, # Backend placeholder
                feature_values=result.get("feature_values")
            )
            
            st.success("✅ Prediction completed successfully!")

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")


# prepare_prediction_input() has been moved to src/core/model.py (ModelManager.prepare_input())


def collect_prediction_inputs():
    """Collect all prediction inputs from the user"""
    
    st.info("💡 **Lưu ý:** Các features phức tạp (lag, rolling stats, EWMA) sẽ được tự động lấy từ dữ liệu lịch sử")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Thông tin ngày")
        DATASET_END = datetime(2014, 7, 31).date()  # Max date in training data
        prediction_date = st.date_input(
            "Ngày dự đoán",
            value=DATASET_END,
            min_value=datetime(2012, 1, 1).date(),
            max_value=datetime(2024, 12, 30).date(),
            help="Giới hạn (2012-01-01 → 2024-12-30) — chọn xa với lịch sử gốc sẽ chạy dự đoán đệ quy lâu hơn"
        )
        is_holiday = st.checkbox("Ngày lễ (Holiday)", value=False)
        is_blackfriday = st.checkbox("Black Friday", value=False)

    with col2:
        st.subheader("🌤️ Thông tin thời tiết")
        tmax = st.slider("Nhiệt độ tối đa (°F)", 0.0, 110.0, 70.0, 0.5)
        cool = st.slider("Cool (Cooling Degree Days)", 0.0, 50.0, 10.0, 0.5)
        preciptotal = st.slider("Lượng mưa (inch)", 0.0, 5.0, 0.0, 0.1)

    # Weather codes
    with st.expander("🌤️ Chọn hiện tượng thời tiết (Weather Codes)"):
        st.caption("Chỉ chọn các mã mà mô hình AI hỗ trợ dự báo.")
        w_col1, w_col2, w_col3 = st.columns(3)
        codes = {}
        
        with w_col1:
            st.markdown("**🌧️ Mưa & Tuyết**")
            codes['RA'] = st.checkbox("Mưa (RA)", help="Rain")
            codes['DZ'] = st.checkbox("Mưa phùn (DZ)", help="Drizzle")
            codes['SN'] = st.checkbox("Tuyết (SN)", help="Snow")
            codes['FZRA'] = st.checkbox("Mưa đóng băng (FZRA)", help="Freezing Rain")
            codes['BLSN'] = st.checkbox("Tuyết thổi (BLSN)", help="Blowing Snow")

        with w_col2:
            st.markdown("**🌫️ Sương mù & Tầm nhìn**")
            codes['FG'] = st.checkbox("Sương mù (FG)", help="Fog")
            codes['BR'] = st.checkbox("Sương mù nhẹ (BR)", help="Mist")
            codes['HZ'] = st.checkbox("Mù khô (HZ)", help="Haze")
            codes['MIFG'] = st.checkbox("Sương mù nông (MIFG)", help="Shallow Fog")
            codes['VCFG'] = st.checkbox("Sương mù lân cận (VCFG)", help="Vicinity Fog")

        with w_col3:
            st.markdown("**⚡ Dông & Gió**")
            codes['TS'] = st.checkbox("Dông (TS)", help="Thunderstorm")
            codes['SQ'] = st.checkbox("Gió giật (SQ)", help="Squalls")
            st.markdown("**Other**")
            codes['PRFG'] = st.checkbox("Sương mù bán phần (PRFG)", help="Partial Fog")
            codes['BCFG'] = st.checkbox("Sương mù từng đám (BCFG)", help="Patchy Fog")
            codes['UP'] = st.checkbox("Ko xác định (UP)", help="Unknown Precipitation")

    st.subheader("🌡️ Áp suất & Gió")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1: stnpressure = st.number_input("Trạm (inHg)", 28.0, 31.0, 29.92)
    with p_col2: sealevel = st.number_input("Biển (inHg)", 28.0, 31.0, 29.92)
    with p_col3: resultspeed = st.number_input("Gió (mph)", 0.0, 50.0, 5.0)
    with p_col4: resultdir = st.number_input("Hướng (độ)", 0, 360, 180)

    with st.expander("🌡️ Thông số thời tiết nâng cao (Advanced)"):
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            tmin = st.number_input("Nhiệt độ tối thiểu (°F)", 0.0, 100.0, 50.0)
            tavg = st.number_input("Nhiệt độ TB (°F)", 0.0, 105.0, 60.0)
            depart = st.number_input("Chênh lệch TB (Depart)", -30.0, 30.0, 0.0)
            snowfall = st.number_input("Tuyết rơi (inch)", 0.0, 20.0, 0.0)
        with a_col2:
            dewpoint = st.number_input("Điểm sương (°F)", -20.0, 90.0, 50.0)
            wetbulb = st.number_input("Nhiệt độ bầu ướt (°F)", -20.0, 90.0, 55.0)
            heat = st.number_input("Heat (Heating Degree Days)", 0.0, 50.0, 0.0)
        with a_col3:
            sunrise = st.number_input("Bình minh (vd 530 = 5:30)", 0.0, 2400.0, 500.0)
            sunset = st.number_input("Hoàng hôn (vd 1830 = 18:30)", 0.0, 2400.0, 1800.0)
            avgspeed = st.number_input("Tốc độ gió TB (mph)", 0.0, 50.0, 5.0)

    # Derived params
    month, day, year = prediction_date.month, prediction_date.day, prediction_date.year
    day_of_week = prediction_date.weekday()
    
    if month in [3, 4, 5]: season = "Spring"
    elif month in [6, 7, 8]: season = "Summer"
    elif month in [12, 1, 2]: season = "Winter"
    else: season = "Fall"

    inputs = {
        "date": prediction_date.isoformat(),
        "year": year, "month": month, "day": day, "day_of_week": day_of_week,
        "is_weekend": 1 if day_of_week >= 5 else 0, "season": season,
        "is_holiday": int(is_holiday), "is_blackfriday": int(is_blackfriday),
        "tmax": tmax, "tmin": tmin, "tavg": tavg, "depart": depart,
        "dewpoint": dewpoint, "wetbulb": wetbulb, "heat": heat, "cool": cool,
        "sunrise": sunrise, "sunset": sunset, "snowfall": snowfall,
        "preciptotal": preciptotal, "stnpressure": stnpressure, "sealevel": sealevel,
        "resultspeed": resultspeed, "resultdir": resultdir, "avgspeed": avgspeed,
        **{c: int(v) for c, v in codes.items()},
        "_date_obj": prediction_date
    }
    return inputs


def display_prediction_results(
    api_client,
    prediction_value,
    store_id,
    item_id,
    prediction_inputs,
    historical_df,
    forecast_history=None,
    feature_values=None
):
    """Display prediction results with visualizations"""
    st.header("Prediction Results")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(label="Predicted Units", value=f"{prediction_value:,.0f}")
        st.write(f"**Store ID:** {store_id}")
        st.write(f"**Product ID:** {item_id}")
        
        date_obj = prediction_inputs.get("_date_obj")
        if date_obj:
            st.write(f"**Date:** {date_obj.strftime('%B %d, %Y')}")
        
        st.write(f"**Season:** {prediction_inputs.get('season')}")

    with res_col2:
        if not historical_df.empty:
            units_col = "units" if "units" in historical_df.columns else "sales"
            avg_units = historical_df[units_col].mean()
            max_units = historical_df[units_col].max()
            
            st.metric(label="Historical Average", value=f"{avg_units:,.0f} units")
            st.metric(label="Historical Maximum", value=f"{max_units:,.0f} units")
            st.write(f"**Period:** {historical_df['date'].min().date()} to {historical_df['date'].max().date()}")

    # Visualizations
    display_historical_context(historical_df, prediction_inputs.get("_date_obj"), prediction_value, forecast_history)

    # Giải thích dự đoán thông qua SHAP (từ API backend)
    st.subheader("Key Factors Influencing This Prediction (SHAP)")
    with st.spinner("Đang tính toán Feature Importance từ Backend..."):
        date_str = prediction_inputs.get("date")
        if date_str and "T" in date_str:
            date_str = date_str.split("T")[0]
        
        try:
            explanation = api_client.get_local_explanation(store_id, item_id, date=date_str, top_n=8, features=feature_values)
            if explanation and "features" in explanation:
                display_feature_importance_api(explanation["features"])
            else:
                st.info("ℹ️ Feature importance not available for this prediction.")
        except Exception as e:
            st.warning(f"⚠️ Could not load explanation: {str(e)}")


def display_historical_context(historical_data, prediction_date, prediction_value, forecast_history=None):
    """Display historical context visualizations"""

    st.subheader("Recent Sales History")

    # Use 'units' column instead of 'sales'
    units_col = "units" if "units" in historical_data.columns else "sales"
    
    if units_col not in historical_data.columns or historical_data.empty:
        st.info(
            "No historical sales data available for this product-store combination."
        )
        return

    # Limit to last 2 months
    last_date = historical_data["date"].max()
    two_months_ago = last_date - pd.Timedelta(days=60)
    recent_history = historical_data[historical_data["date"] >= two_months_ago]

    if recent_history.empty:
        st.info("No recent sales data available for the last 60 days.")
        return

    # Plot recent sales history - ADJUSTED SIZE
    fig, ax = plt.subplots(figsize=(10, 3.5))  # Wider to accommodate forecast

    # Plot historical units
    ax.plot(
        recent_history["date"],
        recent_history[units_col],
        "b-",
        label="Historical Units",
        linewidth=2,
    )

    # Plot forecast history if available
    if forecast_history is not None and len(forecast_history) > 0:
        ax.plot(
            forecast_history["date"],
            forecast_history["predicted_units"],
            "orange",
            linestyle="--",
            label=f"Recursive Forecast ({len(forecast_history)} days)",
            linewidth=2,
            alpha=0.8,
        )

    # Add the final prediction point
    ax.scatter(
        prediction_date,
        prediction_value,
        color="red",
        s=100,
        label="Final Prediction",
        zorder=5,
        edgecolors='black',
        linewidths=1.5,
    )

    # Add moving average for historical data
    if len(recent_history) > 7:
        recent_history_copy = recent_history.copy()
        recent_history_copy["MA7"] = recent_history_copy[units_col].rolling(window=7).mean()
        ax.plot(
            recent_history_copy["date"],
            recent_history_copy["MA7"],
            "g--",
            label="7-Day Avg",
            alpha=0.6,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Units Sold")
    title = f"Sales History + Recursive Forecast" if forecast_history is not None else "Last 60 Days Sales History"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    st.pyplot(fig)

    # Show forecast statistics if available
    if forecast_history is not None and len(forecast_history) > 0:
        st.caption(f"✨ Recursive forecast: {len(forecast_history)} daily predictions from {forecast_history['date'].min().date()} to {forecast_history['date'].max().date()}")

    # Weekly pattern visualization
    display_weekly_pattern(recent_history, prediction_date, units_col)


def display_weekly_pattern(recent_history, prediction_date, units_col="units"):
    """Display weekly sales pattern visualization"""

    if len(recent_history) >= 7:
        st.subheader("Weekly Sales Pattern")

        # Add day of week
        recent_history["day_of_week"] = recent_history["date"].dt.dayofweek
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        # Group by day of week
        day_units = recent_history.groupby("day_of_week")[units_col].mean()
        day_units_df = pd.DataFrame(
            {
                "day_name": [day_names[i] for i in range(7) if i in day_units.index],
                "units": [day_units[i] for i in range(7) if i in day_units.index],
            }
        )

        # Plot - SMALLER SIZE
        fig, ax = plt.subplots(figsize=(6, 2.5))  # Reduced size

        # Plot day of week pattern
        sns.barplot(x="day_name", y="units", data=day_units_df, ax=ax)

        # Highlight the day of the prediction
        prediction_day = prediction_date.weekday()
        for i, patch in enumerate(ax.patches):
            if day_units_df.iloc[i]["day_name"] == day_names[prediction_day]:
                patch.set_facecolor("red")

        ax.set_xlabel("")
        ax.set_ylabel("Avg Units")
        ax.set_title("Units Sold by Day of Week")
        plt.xticks(rotation=45, fontsize=8)  # Smaller font
        fig.tight_layout()

        st.pyplot(fig)


def display_feature_importance_api(features_data):
    """Display feature importance visualization using XAI Data API"""
    if not features_data:
        return
        
    # Tạo DataFrame từ JSON [{'feature': '..', 'value': .., 'shap_impact': ...}, ...]
    importance_df = pd.DataFrame(features_data)
    
    if "shap_impact" not in importance_df.columns:
        return
        
    # Lấy giá trị tuyệt đối để sắp xếp độ quan trọng (tác động lên hay xuống đều quan trọng)
    importance_df["Absolute Impact"] = importance_df["shap_impact"].abs()
    importance_df = importance_df.sort_values("Absolute Impact", ascending=False).head(8)
    
    # Format lại tên cho đẹp
    importance_df["Feature"] = importance_df["feature"].apply(
        lambda x: x.replace("_", " ").title()
    )

    # Plot
    fig, ax = plt.subplots(figsize=(6, 2.5))
    
    # Đổi màu xanh/đỏ tùy theo tác động âm hay dương
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in importance_df['shap_impact']]
    
    sns.barplot(x="shap_impact", y="Feature", data=importance_df, ax=ax, palette=colors)
    ax.set_title("Top SHAP Feature Impacts", fontsize=10, fontweight='bold')
    ax.set_xlabel("Impact on Predicted Sales", fontsize=8)
    ax.set_ylabel("")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()

    st.pyplot(fig)
