import pandas as pd
import streamlit as st
from services import get_api_client
from components.ui_builder.data_viz import (
    plot_day_of_week_pattern,
    plot_product_comparison,
    plot_sales_distribution,
    plot_sales_time_series,
    plot_store_comparison,
    plot_products_trend_comparison,
    plot_market_share_pie,
    plot_growth_rate_comparison,
    plot_seasonality_heatmap,
    plot_hist_zoom_tail,
)

def historical_sales_view(data=None):
    """Display the historical sales analysis dashboard via Backend API"""
    st.title("Store Sales Dashboard")
    
    api = get_api_client()
    
    # 1. Dashboard Filters
    filters = configure_filters_api(api)
    if not filters:
        st.error("Could not load dashboard filters.")
        return
        
    st.session_state.update(filters)

    # 2. KPI Section
    display_kpis_api(api, filters)

    # 3. Sales Trends
    display_sales_trends_api(api, filters)

    # 4. Performance Breakdown
    display_performance_breakdown_api(api, filters)

    # 5. Product Performance (Comparative)
    display_product_performance_analysis_api(api, filters)

    # 6. Units Distribution
    display_sales_distribution_api(api, filters)

    # 7. Data Table
    with st.expander("View Detailed Sales Data"):
        sid = None if filters["selected_store"] == "All Stores" else int(filters["selected_store"])
        iid = None if filters["selected_item"] == "All Products" else int(filters["selected_item"])
        
        detail_res = api.get_historical_data(store_id=sid, item_id=iid, limit=1000)
        if detail_res and "data" in detail_res:
            detail_df = pd.DataFrame(detail_res["data"])
            st.dataframe(detail_df, width="stretch")


def inject_kpi_css():
    st.markdown("""
    <link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,500,0,0" />
    <style>
      .kpi-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 12px;
        padding: 14px 16px;
        background: white;
      }
      .kpi-top {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
      }
      .kpi-icon {
        width: 34px; height: 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0,0,0,0.04);
      }
      .material-symbols-outlined { font-size: 22px; line-height: 1; }
      .kpi-label { font-size: 0.85rem; color: rgba(49, 51, 63, 0.7); }
      .kpi-value { font-size: 2.0rem; font-weight: 700; color: rgba(49, 51, 63, 0.95); }
      .kpi-delta { font-size: 0.85rem; margin-top: 4px; }
      .kpi-delta.pos { color: #1B5E20; }
      .kpi-delta.neg { color: #B71C1C; }
      .kpi-delta.neu { color: rgba(49, 51, 63, 0.65); }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, icon_name: str, icon_color: str,
             delta_text: str | None = None, delta_class: str = "neu"):
    """Render a KPI card using HTML/CSS"""
    delta_html = f'<div class="kpi-delta {delta_class}">{delta_text}</div>' if delta_text else ""
    st.markdown(f"""
      <div class="kpi-card">
        <div class="kpi-top">
          <div class="kpi-icon">
            <span class="material-symbols-outlined" style="color:{icon_color};">
              {icon_name}
            </span>
          </div>
          <div class="kpi-label">{label}</div>
        </div>
        <div class="kpi-value">{value}</div>
        {delta_html}
      </div>
    """, unsafe_allow_html=True)


def configure_filters_api(api):
    """Configure dashboard filters via API calls"""
    with st.sidebar:
        st.header("Dashboard Filters")
        
        # Get bounds and stores
        meta = api.get_analytics_filters()
        if "error" in meta:
            return None
            
        min_date = pd.to_datetime(meta["min_date"]).date()
        max_date = pd.to_datetime(meta["max_date"]).date()
        
        # Date Range
        default_start = max_date - pd.Timedelta(days=90)
        start_date = st.date_input("From", default_start, min_value=min_date, max_value=max_date)
        end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
        
        # Store Selector
        store_options = ["All Stores"] + [str(s) for s in meta["stores"]]
        selected_store = st.selectbox("Select Store", options=store_options)
        
        # Cascading Item Selector
        sid = None if selected_store == "All Stores" else int(selected_store)
        items = api.get_analytics_items(store_id=sid)
        item_options = ["All Products"] + [str(i) for i in items]
        selected_item = st.selectbox("Select Product", options=item_options)
        
        selected_store_name = "All Stores" if selected_store == "All Stores" else f"Store {selected_store}"

    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "selected_store": selected_store,
        "selected_store_name": selected_store_name,
        "selected_item": selected_item
    }


def display_kpis_api(api, filters):
    """Display KPI metrics from API"""
    st.header("Key Performance Indicators")
    inject_kpi_css()
    
    sid = None if filters["selected_store"] == "All Stores" else int(filters["selected_store"])
    res = api.get_kpis_data(filters["start_date"], filters["end_date"], store_id=sid)
    
    if "error" in res:
        st.warning("No data for KPIs")
        return

    col1, col2, col3 = st.columns(3)
    
    with col1:
        growth = res.get("growth_pct", 0)
        diff = res.get("p2_units", 0) - res.get("p1_units", 0)
        delta_class = "pos" if growth > 0 else ("neg" if growth < 0 else "neu")
        kpi_card(
            label="Total Units Sold",
            value=f"{res.get('total_units', 0):,.0f}",
            icon_name="inventory_2",
            icon_color="#4CAF50",
            delta_text=f"vs 1st half: {diff:+.0f} ({growth:+.1f}%)",
            delta_class=delta_class,
        )
            
    with col2:
        kpi_card(
            label="Avg Daily Units",
            value=f"{res.get('avg_daily', 0):,.1f}",
            icon_name="insights",
            icon_color="#2196F3",
            delta_text=f"Avg/day over {res.get('days_count', 1):,} days",
            delta_class="neu"
        )
            
    with col3:
        records_per_day = res.get("total_records", 0) / res.get("days_count", 1)
        kpi_card(
            label="Sales Records (proxy)",
            value=f"{res.get('total_records', 0):,}",
            icon_name="receipt_long",
            icon_color="#FF9800",
            delta_text=f"~{records_per_day:,.0f} records/day",
            delta_class="neu"
        )


def display_sales_trends_api(api, filters):
    """Display trends from API"""
    st.header("Sales Trends")
    col1, col2 = st.columns(2)
    
    sid = None if filters["selected_store"] == "All Stores" else int(filters["selected_store"])
    res = api.get_trends_data(filters["start_date"], filters["end_date"], store_id=sid)
    
    if not res.get("data"):
        return
        
    trend_df = pd.DataFrame(res["data"])
    trend_df["date"] = pd.to_datetime(trend_df["date"])
    
    with col1:
        fig = plot_sales_time_series(trend_df, filters["selected_store"], filters["selected_store_name"])
        st.pyplot(fig)
        
    with col2:
        fig_dow = plot_day_of_week_pattern(trend_df)
        st.pyplot(fig_dow)


def display_performance_breakdown_api(api, filters):
    """Performance breakdown from API"""
    st.header("Performance Breakdown")
    col1, col2 = st.columns(2)
    
    sid = None if filters["selected_store"] == "All Stores" else int(filters["selected_store"])
    res = api.get_performance_data(filters["start_date"], filters["end_date"], store_id=sid)
    
    with col1:
        st.subheader("Top 10 Products")
        if res.get("top_items"):
            top_items_df = pd.DataFrame(res["top_items"])
            st.dataframe(top_items_df, width="stretch")
            st.pyplot(plot_product_comparison(top_items_df, "item_id"))
            
    with col2:
        st.subheader("Top 10 Stores")
        if res.get("top_stores"):
            top_stores_df = pd.DataFrame(res["top_stores"])
            st.dataframe(top_stores_df, width="stretch")
            st.pyplot(plot_store_comparison(top_stores_df, "store_id"))


def display_product_performance_analysis_api(api, filters):
    """Comparative analysis from API"""
    sid = None if filters["selected_store"] == "All Stores" else int(filters["selected_store"])
    items = api.get_analytics_items(store_id=sid)
    
    if len(items) < 2: return

    st.markdown("---")
    st.header("🎯 Product Performance Analysis")
    
    selected_products = st.multiselect(
        "Select Products to Compare",
        options=items,
        default=[i for i in [44, 43, 38] if i in items],
        max_selections=5
    )

    if len(selected_products) < 2:
        st.info("Please select at least 2 products.")
        return

    res = api.get_comparison_data(filters["start_date"], filters["end_date"], item_ids=selected_products, store_id=sid)
    if not res.get("data"):
        return
        
    comp_df = pd.DataFrame(res["data"])
    comp_df["date"] = pd.to_datetime(comp_df["date"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Sales Trend Comparison")
        st.pyplot(plot_products_trend_comparison(comp_df, selected_products))
    with col2:
        st.subheader("Market Share")
        st.pyplot(plot_market_share_pie(comp_df, selected_products))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Growth Rate Trend")
        st.pyplot(plot_growth_rate_comparison(comp_df, selected_products))
    with col4:
        st.subheader("Seasonality Patterns")
        st.pyplot(plot_seasonality_heatmap(comp_df, selected_products))


def display_sales_distribution_api(api, filters):
    """Distribution analysis from API"""
    st.header("Units Distribution Analysis")
    st.markdown("Explore distribution at different grains to understand sales patterns.")
    
    sid = None if filters["selected_store"] == "All Stores" else int(filters["selected_store"])
    iid = None if filters["selected_item"] == "All Products" else int(filters["selected_item"])
    
    res = api.get_distribution_data(filters["start_date"], filters["end_date"], store_id=sid, item_id=iid)
    
    if res.get("error"):
        st.error("Failed to load distribution data")
        return

    tab1, tab2 = st.tabs(["Item-Store-Day (Detailed)", "Store-Day (Aggregated)"])
    
    # Tab 1: Detailed
    with tab1:
        st.caption("Distribution of individual sales records (1 record = 1 item sold at 1 store on 1 day).")
        data1 = res.get("detailed", [])
        if data1:
            df1 = pd.DataFrame({"units": data1})
            st.pyplot(plot_hist_zoom_tail(df1, "units", p=0.99))
        else:
            st.info("No detailed data available.")

    # Tab 2: Aggregated
    with tab2:
        st.caption("Distribution of total daily sales per store. Indicates store busyness.")
        data2 = res.get("aggregated", [])
        if data2:
            df2 = pd.DataFrame({"store_day_units": data2})
            st.pyplot(plot_hist_zoom_tail(df2, "store_day_units", p=0.99))
        else:
            st.info("No aggregated data available.")
