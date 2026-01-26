import pandas as pd
import streamlit as st
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
from utils.db_manager import run_query

def historical_sales_view(data=None):
    """Display the historical sales analysis dashboard using dbt marts"""

    st.title("Store Sales Dashboard")

    # Dashboard Filters Section (Now using SQL)
    filters = configure_filters_sql()
    
    # Store settings in session state for consistency
    st.session_state.update(filters)

    # 1. KPI Section
    display_kpis_sql(filters)

    # 2. Sales Trends
    display_sales_trends_sql(filters)

    # 3. Performance Breakdown
    display_performance_breakdown_sql(filters)

    # 4. Product Performance (Comparative)
    display_product_performance_analysis_sql(filters)

    # 5. Units Distribution
    display_sales_distribution_sql(filters)

    # 6. Data Table
    with st.expander("View Detailed Sales Data"):
        query = """
            SELECT * FROM mart_sales_base 
            WHERE date BETWEEN :start_date AND :end_date
        """
        params = {"start_date": filters["start_date"], "end_date": filters["end_date"]}
        
        if filters["selected_store"] != "All Stores":
            query += " AND store_id = :store_id"
            params["store_id"] = filters["selected_store"]
            
        if filters["selected_item"] != "All Products":
            query += " AND item_id = :item_id"
            params["item_id"] = filters["selected_item"]
            
        query += " ORDER BY date DESC LIMIT 1000"
        
        detail_df = run_query(query, params)
        st.dataframe(detail_df, width="stretch")


def configure_filters_sql():
    """Configure dashboard filters using SQL queries (Cascading)"""
    
    with st.sidebar:
        st.header("Dashboard Filters")
        
        # 1. Date Range
        date_bounds = run_query("SELECT MIN(date) as min_d, MAX(date) as max_d FROM mart_date_sales")
        min_date = date_bounds["min_d"].iloc[0]
        max_date = date_bounds["max_d"].iloc[0]
        
        # Default to last 90 days
        default_start = max_date - pd.Timedelta(days=90)
        if default_start < min_date:
            default_start = min_date

        start_date = st.date_input("From", default_start, min_value=min_date, max_value=max_date)
        end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
        
        # 2. Store Selector
        stores_df = run_query("SELECT DISTINCT store_id FROM mart_store_day ORDER BY store_id")
        store_options = ["All Stores"] + stores_df["store_id"].astype(str).tolist()
        selected_store = st.selectbox("Select Store", options=store_options)
        
        # 3. Cascading Item Selector (User logic step 3)
        if selected_store == "All Stores":
            items_query = "SELECT DISTINCT item_id FROM mart_sales_base ORDER BY item_id"
            items_df = run_query(items_query)
        else:
            items_query = "SELECT DISTINCT item_id FROM mart_sales_base WHERE store_id = :s_id ORDER BY item_id"
            items_df = run_query(items_query, {"s_id": int(selected_store)})
            
        item_options = ["All Products"] + items_df["item_id"].astype(str).tolist()
        selected_item = st.selectbox("Select Product", options=item_options)
        
        # Helper for display names
        selected_store_name = "All Stores" if selected_store == "All Stores" else f"Store {selected_store}"

    return {
        "start_date": start_date,
        "end_date": end_date,
        "selected_store": selected_store,
        "selected_store_name": selected_store_name,
        "selected_item": selected_item
    }


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


def display_kpis_sql(filters):
    """Display KPI metrics querying marts with period comparison"""
    st.header("Key Performance Indicators")
    
    # Inject CSS for KPI cards
    inject_kpi_css()
    
    # Calculate mid-point for growth comparison
    # We query the actual range from DB to be precise
    if filters["selected_store"] == "All Stores":
        dates_q = "SELECT MIN(date), MAX(date) FROM mart_date_sales WHERE date BETWEEN :s AND :e"
        grain_table = "mart_date_sales"
        where_clause = "date BETWEEN :start AND :end"
        qp = {"s": filters["start_date"], "e": filters["end_date"]}
        
    else:
        dates_q = "SELECT MIN(date), MAX(date) FROM mart_store_day WHERE store_id = :sid AND date BETWEEN :s AND :e"
        grain_table = "mart_store_day"
        where_clause = "store_id = :s_id AND date BETWEEN :start AND :end"
        qp = {"sid": int(filters["selected_store"]), "s": filters["start_date"], "e": filters["end_date"]}
        
    dates = run_query(dates_q, qp)
    if dates.iloc[0,0] is None:
        st.warning("No data for KPIs")
        return

    min_d, max_d = dates.iloc[0,0], dates.iloc[0,1]
    mid_d = min_d + (max_d - min_d) / 2
    
    # Combined query for totals and period splits
    query = f"""
        SELECT 
            SUM(total_units) as total_units,
            AVG(total_units) as avg_daily,
            SUM(sales_records) as total_records,
            SUM(CASE WHEN date <= :mid THEN total_units ELSE 0 END) as p1_units,
            SUM(CASE WHEN date > :mid THEN total_units ELSE 0 END) as p2_units
        FROM {grain_table}
        WHERE {where_clause}
    """
    
    params = {
        "start": filters["start_date"], 
        "end": filters["end_date"],
        "mid": mid_d
    }
    if filters["selected_store"] != "All Stores":
        params["s_id"] = int(filters["selected_store"])

    kpi_df = run_query(query, params)
    
    if kpi_df.empty:
        return

    total_units = kpi_df["total_units"].iloc[0] or 0
    avg_daily = kpi_df["avg_daily"].iloc[0] or 0
    total_records = kpi_df["total_records"].iloc[0] or 0
    p1 = kpi_df["p1_units"].iloc[0] or 0
    p2 = kpi_df["p2_units"].iloc[0] or 0
    
    # Growth calculation
    if p1 > 0:
        growth_pct = ((p2 - p1) / p1) * 100
        # Check if period 2 is incomplete or much shorter? 
        # Simple period-over-period for now.
    else:
        growth_pct = 0.0

    # Calculate day count for Avg Daily explanation
    days_count_q = f"SELECT COUNT(DISTINCT date) FROM {grain_table} WHERE {where_clause}"
    days_count = run_query(days_count_q, params).iloc[0,0] or 1

    col1, col2, col3 = st.columns(3)
    
    # 1. Total Units with better PoP context
    diff = p2 - p1
    delta_class = "pos" if growth_pct > 0 else ("neg" if growth_pct < 0 else "neu")
    delta_text = f"vs 1st half: {diff:+.0f} ({growth_pct:+.1f}%)"

    with col1:
        kpi_card(
            label="Total Units Sold",
            value=f"{total_units:,.0f}",
            icon_name="inventory_2",
            icon_color="#4CAF50",
            delta_text=delta_text,
            delta_class=delta_class,
        )
            
    # 2. Avg Daily with explanation
    avg_explanation = f"Avg/day over {days_count:,} days"
    with col2:
        kpi_card(
            label="Avg Daily Units",
            value=f"{avg_daily:,.1f}",
            icon_name="insights",
            icon_color="#2196F3",
            delta_text=avg_explanation,
            delta_class="neu"
        )
            
    # 3. Records with density context
    records_per_day = total_records / days_count if days_count > 0 else 0
    density_text = f"~{records_per_day:,.0f} records/day"
    with col3:
        kpi_card(
            label="Sales Records (proxy)",
            value=f"{total_records:,}",
            icon_name="receipt_long",
            icon_color="#FF9800",
            delta_text=density_text,
            delta_class="neu"
        )


def display_sales_trends_sql(filters):
    """Display trends using SQL series"""
    st.header("Sales Trends")
    col1, col2 = st.columns(2)
    
    # 1. Time Series
    if filters["selected_store"] == "All Stores":
        query = "SELECT date, total_units as units FROM mart_date_sales WHERE date BETWEEN :s AND :e ORDER BY date"
        params = {"s": filters["start_date"], "e": filters["end_date"]}
    else:
        query = "SELECT date, total_units as units FROM mart_store_day WHERE store_id = :id AND date BETWEEN :s AND :e ORDER BY date"
        params = {"id": int(filters["selected_store"]), "s": filters["start_date"], "e": filters["end_date"]}
    
    trend_df = run_query(query, params)
    trend_df["date"] = pd.to_datetime(trend_df["date"])
    
    with col1:
        # Mocking the call expected by plot_sales_time_series
        # The plot function expects 'filtered_data' with 'units' and 'date'
        fig = plot_sales_time_series(trend_df, filters["selected_store"], filters["selected_store_name"])
        st.pyplot(fig)
        
    # 2. Day of Week (SQL Groupby)
    query_dow = """
        SELECT day_of_week, AVG(total_units) as units 
        FROM mart_date_sales 
        WHERE date BETWEEN :s AND :e
        GROUP BY 1 ORDER BY 1
    """
    dow_df = run_query(query_dow, {"s": filters["start_date"], "e": filters["end_date"]})
    
    # Re-map day of week to names for plot_day_of_week_pattern or helper
    with col2:
        if not trend_df.empty:
            # We pass the full daily data to the plotter which handles mapping
            fig_dow = plot_day_of_week_pattern(trend_df)
            st.pyplot(fig_dow)


def display_performance_breakdown_sql(filters):
    """Top 10 items/stores breakdown"""
    st.header("Performance Breakdown")
    
    col1, col2 = st.columns(2)
    
    # Top 10 Items
    with col1:
        st.subheader("Top 10 Products")
        q_items = """
            SELECT item_id as item_nbr, SUM(units) as units 
            FROM mart_sales_base 
            WHERE date BETWEEN :s AND :e
        """
        p_items = {"s": filters["start_date"], "e": filters["end_date"]}
        if filters["selected_store"] != "All Stores":
            q_items += " AND store_id = :sid"
            p_items["sid"] = int(filters["selected_store"])
        q_items += " GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
        
        top_items_df = run_query(q_items, p_items)
        if not top_items_df.empty:
            st.dataframe(top_items_df, width="stretch")
            fig = plot_product_comparison(top_items_df, "item_nbr")
            st.pyplot(fig)
            
    # Top 10 Stores
    with col2:
        st.subheader("Top 10 Stores")
        q_stores = """
            SELECT store_id as store_nbr, SUM(total_units) as units 
            FROM mart_store_day 
            WHERE date BETWEEN :s AND :e
            GROUP BY 1 ORDER BY 2 DESC LIMIT 10
        """
        top_stores_df = run_query(q_stores, {"s": filters["start_date"], "e": filters["end_date"]})
        if not top_stores_df.empty:
            st.dataframe(top_stores_df, width="stretch")
            fig = plot_store_comparison(top_stores_df, "store_nbr")
            st.pyplot(fig)


def display_product_performance_analysis_sql(filters):
    """Detailed benchmarking for selected products"""
    # Quick check for product existence
    q_all = """
        SELECT DISTINCT item_id FROM mart_sales_base 
        WHERE date BETWEEN :s AND :e
    """
    p_all = {"s": filters["start_date"], "e": filters["end_date"]}
    if filters["selected_store"] != "All Stores":
        q_all += " AND store_id = :sid"
        p_all["sid"] = int(filters["selected_store"])
        
    available_items = run_query(q_all, p_all)["item_id"].tolist()
    
    if len(available_items) < 2: return

    st.markdown("---")
    st.header("🎯 Product Performance Analysis")
    
    selected_products = st.multiselect(
        "Select Products to Compare",
        options=available_items,
        default=[i for i in [44, 43, 38] if i in available_items],
        max_selections=5
    )

    if len(selected_products) < 2:
        st.info("Please select at least 2 products.")
        return

    q_comp = """
        SELECT date, item_id as item_nbr, units 
        FROM mart_sales_base 
        WHERE item_id = ANY(:items) AND date BETWEEN :s AND :e
    """
    params = {"items": list(selected_products), "s": filters["start_date"], "e": filters["end_date"]}
    comp_df = run_query(q_comp, params)
    comp_df["date"] = pd.to_datetime(comp_df["date"])

    # Plots
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Sales Trend Comparison")
        st.pyplot(plot_products_trend_comparison(comp_df, selected_products))
    with col2:
        st.subheader("Market Share")
        # For market share, we need total store sales to show "Others"
        total_q = "SELECT SUM(total_units) FROM mart_date_sales WHERE date BETWEEN :s AND :e"
        total_s = run_query(total_q, {"s": filters["start_date"], "e": filters["end_date"]}).iloc[0,0]
        st.pyplot(plot_market_share_pie(comp_df, selected_products))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Growth Rate Trend")
        st.pyplot(plot_growth_rate_comparison(comp_df, selected_products))
    with col4:
        st.subheader("Seasonality Patterns")
        st.pyplot(plot_seasonality_heatmap(comp_df, selected_products))


def display_sales_distribution_sql(filters):
    """Histogram of sales units (Granular vs Aggregated)"""
    st.header("Units Distribution Analysis")
    st.markdown("Explore distribution at different grains to understand sales patterns.")

    tab1, tab2 = st.tabs(["Item-Store-Day (Detailed)", "Store-Day (Aggregated)"])

    # Tab 1: Item-Store-Day (Original Granularity)
    with tab1:
        st.caption("Distribution of individual sales records (1 record = 1 item sold at 1 store on 1 day). Shows granularity of sales.")
        q1 = """
            SELECT units 
            FROM mart_sales_base
            WHERE date BETWEEN :s AND :e
        """
        params1 = {"s": filters["start_date"], "e": filters["end_date"]}
        
        if filters["selected_store"] != "All Stores":
            q1 += " AND store_id = :sid"
            params1["sid"] = int(filters["selected_store"])
            
        if filters.get("selected_item") and filters["selected_item"] != "All Products":
            q1 += " AND item_id = :iid"
            params1["iid"] = int(filters["selected_item"])
            
        df1 = run_query(q1, params1)
        if not df1.empty:
            st.pyplot(plot_hist_zoom_tail(df1, "units", p=0.99))
        else:
            st.info("No data available.")

    # Tab 2: Store-Day (Aggregated)
    with tab2:
        st.caption("Distribution of total daily sales per store. Smoother and indicates store busyness.")
        q2 = """
            SELECT
              date,
              store_id,
              SUM(units) AS store_day_units
            FROM mart_sales_base
            WHERE date BETWEEN :s AND :e
        """
        params2 = {"s": filters["start_date"], "e": filters["end_date"]}

        if filters["selected_store"] != "All Stores":
            q2 += " AND store_id = :sid"
            params2["sid"] = int(filters["selected_store"])

        if filters.get("selected_item") and filters["selected_item"] != "All Products":
            q2 += " AND item_id = :iid"
            params2["iid"] = int(filters["selected_item"])

        q2 += " GROUP BY 1,2"

        df2 = run_query(q2, params2)
        if not df2.empty:
            st.pyplot(plot_hist_zoom_tail(df2, "store_day_units", p=0.99))
        else:
            st.info("No data available.")
