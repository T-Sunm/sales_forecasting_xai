from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from services import get_api_client
from utils.data_helpers import get_top_data_pair


def sales_prediction_view(data, api_client, feature_stats, feature_engineered_data):
    """
    Display the sales prediction tool interface
    
    Args:
        data: Historical data for visualization
        api_client: APIClient instance for backend communication
        feature_stats: Feature statistics dictionary
        feature_engineered_data: Full feature-engineered dataset
    """
    st.title("Sales Prediction Tool")

    if feature_engineered_data.empty:
        st.error("Feature engineered data not loaded.")
        return

    # Determine store and item column names
    if "store_nbr" in feature_engineered_data.columns:
        store_col = "store_nbr"
    
    if "item_nbr" in feature_engineered_data.columns:
        item_col = "item_nbr"

    # Check for store/item name columns
    has_store_names = "store_name" in feature_engineered_data.columns
    has_item_names = "item_name" in feature_engineered_data.columns

    # Create mapping dictionaries for names if available
    store_names, item_names = create_name_mappings(
        feature_engineered_data, store_col, item_col, has_store_names, has_item_names
    )

    # Get unique store and item lists
    stores = sorted(feature_engineered_data[store_col].unique())

    # Create sidebar for selections
    store_id, item_id = create_product_selection_sidebar(
        feature_engineered_data,
        stores,
        store_col,
        item_col,
        has_store_names,
        has_item_names,
        store_names,
        item_names,
    )

    # Main form content
    st.subheader("Prediction Parameters")

    prediction_inputs = collect_prediction_inputs()

    # Make prediction button
    if st.button("Predict Sales", type="primary"):
        generate_prediction(
            api_client,
            feature_engineered_data,
            store_id,
            item_id,
            store_col,
            item_col,
            prediction_inputs,
            has_store_names,
            has_item_names,
            store_names,
            item_names,
        )


def create_name_mappings(df, store_col, item_col, has_store_names, has_item_names):
    """Create mapping dictionaries for store and item names"""

    store_names = {}
    item_names = {}

    if has_store_names:
        # Create store ID to name mapping
        for _, row in df[[store_col, "store_name"]].drop_duplicates().iterrows():
            store_names[row[store_col]] = row["store_name"]

    if has_item_names:
        # Create item ID to name mapping
        for _, row in df[[item_col, "item_name"]].drop_duplicates().iterrows():
            item_names[row[item_col]] = row["item_name"]

    return store_names, item_names


def create_product_selection_sidebar(
    df,
    stores,
    store_col,
    item_col,
    has_store_names,
    has_item_names,
    store_names,
    item_names,
):
    """Create sidebar for store and product selection"""

    # Get top pair for default selection
    top_store, top_item = get_top_data_pair(df)
    
    with st.sidebar:
        st.header("Product Selection")

        # Store selection with names if available
        if has_store_names:
            store_options = [
                f"{store_id} - {store_names[store_id]}" for store_id in stores
            ]
            # Find index of top store
            default_store_idx = 0
            if top_store is not None:
                for i, s in enumerate(stores):
                    if s == top_store:
                        default_store_idx = i
                        break
                        
            selected_store_option = st.selectbox("Select Store", options=store_options, index=default_store_idx)
            store_id = int(selected_store_option.split(" - ")[0])
        else:
            # Find index of top store
            default_store_idx = 0
            if top_store is not None and top_store in stores:
                default_store_idx = stores.index(top_store)
                
            store_id = st.selectbox("Select Store ID", options=stores, index=default_store_idx)

        # Get items for the selected store
        store_items = sorted(df[df[store_col] == store_id][item_col].unique())

        # Item selection with names if available
        if has_item_names:
            item_options = [
                f"{item_id} - {item_names[item_id]}"
                for item_id in store_items
                if item_id in item_names
            ]
            
            # Find index of top item if it's in the current store's items
            default_item_idx = 0
            if top_item is not None and top_item in store_items:
                # We need to find the index in the options list
                for i, opt in enumerate(item_options):
                    if opt.startswith(f"{top_item} -"):
                        default_item_idx = i
                        break
            
            selected_item_option = st.selectbox("Select Product", options=item_options, index=default_item_idx)
            item_id = int(selected_item_option.split(" - ")[0])
        else:
            # Find index of top item if it's in the current store's items
            default_item_idx = 0
            if top_item is not None and top_item in store_items:
                default_item_idx = store_items.index(top_item)
                
            item_id = st.selectbox("Select Product ID", options=store_items, index=default_item_idx)

    return store_id, item_id


def collect_prediction_inputs():
    """Collect all prediction inputs from the user"""
    
    st.info("💡 **Lưu ý:** Các features phức tạp (lag, rolling stats, EWMA) sẽ được tự động lấy từ dữ liệu lịch sử")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Thông tin ngày")
        # Date selection
        prediction_date = st.date_input(
            "Ngày dự đoán", 
            datetime.now().date() + timedelta(days=1)
        )

        # Holiday checkbox
        is_holiday = st.checkbox("Ngày lễ (Holiday)", value=False)
        
        # Black Friday checkbox
        is_blackfriday = st.checkbox("Black Friday", value=False)

    with col2:
        st.subheader("🌤️ Thông tin thời tiết")
        
        # Temperature (tmax in the features)
        tmax = st.slider("Nhiệt độ tối đa (°F)", 0.0, 110.0, 70.0, 0.5)
        
        # Cool (cooling degree days)
        cool = st.slider("Cool (Cooling Degree Days)", 0.0, 50.0, 10.0, 0.5)
        
        # Precipitation
        preciptotal = st.slider("Lượng mưa (inch)", 0.0, 5.0, 0.0, 0.1)

    # Weather section
    st.subheader("🌦️ Điều kiện thời tiết chi tiết")
    
    # Create expandable section for weather codes
    with st.expander("Chọn các hiện tượng thời tiết (Weather Codes)", expanded=False):
        st.caption("Chọn các hiện tượng thời tiết đang xảy ra hoặc dự kiến xảy ra:")
        
        # Weather codes organized by category
        weather_col1, weather_col2, weather_col3 = st.columns(3)
        
        weather_codes = {}
        
        with weather_col1:
            st.markdown("**☁️ Mây/Sương mù**")
            weather_codes['FG'] = st.checkbox("FG - Sương mù (Fog)", value=False)
            weather_codes['FG+'] = st.checkbox("FG+ - Sương mù dày", value=False)
            weather_codes['MIFG'] = st.checkbox("MIFG - Sương mù mỏng", value=False)
            weather_codes['PRFG'] = st.checkbox("PRFG - Sương mù từng phần", value=False)
            weather_codes['FZFG'] = st.checkbox("FZFG - Sương mù đóng băng", value=False)
            weather_codes['VCFG'] = st.checkbox("VCFG - Sương mù gần", value=False)
            weather_codes['BR'] = st.checkbox("BR - Sương mù nhẹ (Mist)", value=False)
            weather_codes['HZ'] = st.checkbox("HZ - Khói mù (Haze)", value=False)
            
        with weather_col2:
            st.markdown("**🌧️ Mưa/Tuyết**")
            weather_codes['RA'] = st.checkbox("RA - Mưa (Rain)", value=False)
            weather_codes['DZ'] = st.checkbox("DZ - Mưa phùn (Drizzle)", value=False)
            weather_codes['FZRA'] = st.checkbox("FZRA - Mưa đóng băng", value=False)
            weather_codes['FZDZ'] = st.checkbox("FZDZ - Mưa phùn đóng băng", value=False)
            weather_codes['SN'] = st.checkbox("SN - Tuyết (Snow)", value=False)
            weather_codes['BLSN'] = st.checkbox("BLSN - Tuyết thổi", value=False)
            weather_codes['SG'] = st.checkbox("SG - Hạt tuyết", value=False)
            weather_codes['PL'] = st.checkbox("PL - Mưa đá nhỏ", value=False)
            weather_codes['GR'] = st.checkbox("GR - Mưa đá (Hail)", value=False)
            weather_codes['GS'] = st.checkbox("GS - Mưa đá nhỏ", value=False)
            
        with weather_col3:
            st.markdown("**⛈️ Dông/Bão bụi**")
            weather_codes['TS'] = st.checkbox("TS - Dông (Thunderstorm)", value=False)
            weather_codes['TSRA'] = st.checkbox("TSRA - Dông có mưa", value=False)
            weather_codes['TSSN'] = st.checkbox("TSSN - Dông có tuyết", value=False)
            weather_codes['VCTS'] = st.checkbox("VCTS - Dông gần", value=False)
            weather_codes['SQ'] = st.checkbox("SQ - Giông (Squall)", value=False)
            weather_codes['DU'] = st.checkbox("DU - Bụi (Dust)", value=False)
            weather_codes['BLDU'] = st.checkbox("BLDU - Bão bụi", value=False)
            weather_codes['FU'] = st.checkbox("FU - Khói (Smoke)", value=False)
            weather_codes['BCFG'] = st.checkbox("BCFG - Sương mù từng mảng", value=False)
            weather_codes['UP'] = st.checkbox("UP - Không xác định", value=False)

    # Atmospheric pressure section
    st.subheader("🌡️ Áp suất khí quyển")
    pressure_col1, pressure_col2 = st.columns(2)
    
    with pressure_col1:
        stnpressure = st.slider("Áp suất trạm (inHg)", 28.0, 31.0, 29.92, 0.01)
    
    with pressure_col2:
        sealevel = st.slider("Áp suất mực nước biển (inHg)", 28.0, 31.0, 29.92, 0.01)

    # Wind section
    st.subheader("💨 Gió")
    wind_col1, wind_col2 = st.columns(2)
    
    with wind_col1:
        resultspeed = st.slider("Tốc độ gió (mph)", 0.0, 50.0, 5.0, 0.5)
    
    with wind_col2:
        resultdir = st.slider("Hướng gió (độ)", 0, 360, 180, 10)

    # Calculate derived parameters
    month = prediction_date.month
    day = prediction_date.day
    year = prediction_date.year
    day_of_week = prediction_date.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Season mapping (matching your features: season_Spring, season_Summer, season_Winter)
    # Note: Fall seems to be the baseline (not in features)
    if month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    elif month in [12, 1, 2]:
        season = "Winter"
    else:
        season = "Fall"  # Baseline

    # Return dictionary for API call (not Pydantic model)
    prediction_dict = {
        "date": prediction_date.isoformat(),
        "year": year,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "season": season,
        "is_holiday": int(is_holiday),
        "is_blackfriday": int(is_blackfriday),
        # Weather features
        "tmax": tmax,
        "cool": cool,
        "preciptotal": preciptotal,
        "stnpressure": stnpressure,
        "sealevel": sealevel,
        "resultspeed": resultspeed,
        "resultdir": resultdir,
        # Weather codes
        **{code: int(value) for code, value in weather_codes.items()},
    }
    
    # Also attach original date object for display purposes
    prediction_dict["_date_obj"] = prediction_date
    
    return prediction_dict


def generate_prediction(
    api_client,
    feature_engineered_data,
    store_id,
    item_id,
    store_col,
    item_col,
    prediction_inputs,
    has_store_names,
    has_item_names,
    store_names,
    item_names,
):
    """
    Generate sales prediction using Backend API and display results
    
    Args:
        api_client: APIClient instance
        feature_engineered_data: Full dataset for context
        store_id, item_id: Selected store and item
        prediction_inputs: Dictionary with prediction parameters
        Other args: Display formatting options
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
            
            # Check for errors (check if error VALUE is truthy, not just if key exists)
            if not result or result.get("error"):
                error_msg = result.get("error", "Unknown error") if result else "No response from backend"
                st.error(f"❌ Prediction failed: {error_msg}")
                return
            
            # Extract prediction value
            prediction_value = result.get("prediction_value")
            
            if prediction_value is None:
                st.error("❌ No prediction value returned from backend")
                return
            
            # Restore date object for display
            if date_obj:
                prediction_inputs["_date_obj"] = date_obj
            
            # Display results (no model/features for now - backend doesn't return them)
            display_prediction_results(
                prediction_value,
                store_id,
                item_id,
                prediction_inputs,
                feature_engineered_data,
                store_col,
                item_col,
                has_store_names,
                has_item_names,
                store_names,
                item_names,
                model=None,  # Not available from API
                model_features=None,  # Not available from API
                forecast_history=None,  # Not available from API yet
            )
            
            st.success("✅ Prediction completed successfully!")

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())


# prepare_prediction_input() has been moved to src/core/model.py (ModelManager.prepare_input())


def display_prediction_results(
    prediction_value,
    store_id,
    item_id,
    prediction_inputs,
    historical_data,
    store_col,
    item_col,
    has_store_names,
    has_item_names,
    store_names,
    item_names,
    model,
    model_features,
    forecast_history=None,
):
    """Display prediction results with visualizations"""

    st.header("Prediction Results")

    # Create results in columns
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        # Display prediction with context
        st.metric(label="Predicted Units", value=f"{prediction_value:,.0f}")

        # Display store and item info
        if has_store_names:
            st.write(f"**Store:** {store_names[store_id]}")
        else:
            st.write(f"**Store ID:** {store_id}")

        if has_item_names:
            st.write(f"**Product:** {item_names[item_id]}")
        else:
            st.write(f"**Product ID:** {item_id}")

        # Handle date display (dict vs Pydantic model)
        date_display = prediction_inputs.get("_date_obj") or prediction_inputs.get("date")
        if date_display:
            if isinstance(date_display, str):
                from datetime import datetime
                date_display = datetime.fromisoformat(date_display).date()
            st.write(f"**Date:** {date_display.strftime('%B %d, %Y')}")
        
        season = prediction_inputs.get("season", "Unknown")
        st.write(f"**Season:** {season}")
        
        if prediction_inputs.get("is_holiday"):
            st.write("**Holiday:** Yes")
        if prediction_inputs.get("is_blackfriday"):
            st.write("**Black Friday:** Yes")

    with res_col2:
        # Get historical context
        historical = historical_data[
            (historical_data[store_col] == store_id)
            & (historical_data[item_col] == item_id)
        ].sort_values("date")

        # Use 'units' column instead of 'sales'
        units_col = "units" if "units" in historical.columns else "sales"
        
        if units_col in historical.columns:
            # Calculate key statistics
            last_value = historical[units_col].iloc[-1] if len(historical) > 0 else 0
            last_date = historical["date"].iloc[-1] if len(historical) > 0 else None

            avg_units = historical[units_col].mean()

            max_units = historical[units_col].max()
            max_date = (
                historical.loc[historical[units_col].idxmax(), "date"]
                if len(historical) > 0
                else None
            )

            # Display average and trend with dates
            st.metric(
                label="Historical Average",
                value=f"{avg_units:,.0f} units",
            )
            st.write(
                f"**Period:** {historical['date'].min().strftime('%b %d, %Y')} to {historical['date'].max().strftime('%b %d, %Y')}"
            )

            st.metric(
                label="Last Recorded Units",
                value=f"{last_value:,.0f} units",
            )
            if last_date is not None:
                st.write(f"**Date:** {last_date.strftime('%b %d, %Y')}")

            st.metric(label="Historical Maximum", value=f"{max_units:,.0f} units")
            if max_date is not None:
                st.write(f"**Date:** {max_date.strftime('%b %d, %Y')}")

    # Historical context
    date_for_display = prediction_inputs.get("_date_obj") or prediction_inputs.get("date")
    if isinstance(date_for_display, str):
        from datetime import datetime
        date_for_display = datetime.fromisoformat(date_for_display).date()
    
    display_historical_context(historical, date_for_display, prediction_value, forecast_history)

    # Feature importance (skip if not available from API)
    if model_features is not None and model is not None:
        display_feature_importance(model, model_features)
    else:
        st.info("ℹ️ Feature importance not available via API (backend doesn't return model details yet)")


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


def display_feature_importance(model, model_features):
    """Display feature importance visualization"""

    if hasattr(model, "feature_importances_"):
        st.subheader("Key Factors Influencing This Prediction")

        # Get feature importances
        importances = model.feature_importances_

        # Create DataFrame with feature importances
        importance_df = (
            pd.DataFrame({"Feature": model_features, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(8)
        )

        # Clean feature names for display
        importance_df["Feature"] = importance_df["Feature"].apply(
            lambda x: x.replace("_", " ").title()
        )

        # Plot feature importances - SMALLER SIZE
        fig, ax = plt.subplots(figsize=(6, 2.5))  # Reduced size
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        ax.set_title("Top Factors Influencing Sales Prediction")
        plt.xticks(fontsize=8)  # Smaller font
        plt.yticks(fontsize=8)  # Smaller font
        fig.tight_layout()

        st.pyplot(fig)
