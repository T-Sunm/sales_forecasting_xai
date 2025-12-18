import pandas as pd
import streamlit as st

from src.ui_builder.data_viz import (
    plot_category_distribution,
    plot_day_of_week_pattern,
    plot_product_comparison,
    plot_sales_distribution,
    plot_sales_time_series,
    plot_store_comparison,
)


def historical_sales_view(data):
    """Display the historical sales analysis dashboard"""

    st.title("Store Sales Dashboard")

    if data.empty:
        st.warning("No sales data available. Please check the data file.")
        return

    # Dashboard Filters Section
    filtered_data = configure_filters(data)

    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
        return

    # Display the KPIs section
    display_kpis(filtered_data)

    # Display the sales trends section
    display_sales_trends(filtered_data)

    # Display the performance breakdown section
    display_performance_breakdown(filtered_data)

    # Display the sales distribution section
    st.header("Units Distribution")
    fig = plot_sales_distribution(filtered_data)
    st.pyplot(fig)

    # Data Table (Expandable)
    with st.expander("View Detailed Sales Data"):
        st.dataframe(
            filtered_data.sort_values("date", ascending=False).head(1000),
            use_container_width=True,
        )


def configure_filters(data):
    """Configure and apply dashboard filters"""
    
    with st.sidebar:
        st.header("Dashboard Filters")
        
        # ===== DATE RANGE =====
        st.subheader("Date Range")
        min_date = data["date"].min().date()
        max_date = data["date"].max().date()
        
        start_date = st.date_input(
            "From", min_date, min_value=min_date, max_value=max_date
        )
        end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
        
        # ===== STORE SELECTOR =====
        st.subheader("Store Selection")
        
        # Xác định column nào có sẵn
        if "store_name" in data.columns:
            store_column = "store_name"
            store_values = sorted(data["store_name"].unique())
        elif "store_nbr" in data.columns:
            store_column = "store_nbr"
            store_values = sorted(data["store_nbr"].unique())
        else:
            store_column = None
            store_values = []
        
        if store_column:
            store_options = ["All Stores"] + store_values
            selected_store = st.selectbox("Select Store", options=store_options)
        else:
            selected_store = "All Stores"
        
        # ===== ITEM SELECTOR (CASCADING - QUAN TRỌNG) =====
        st.subheader("Product Selection")
        
        # Lọc items dựa trên store đã chọn
        if selected_store == "All Stores":
            # Hiển thị TẤT CẢ items
            available_items = sorted(data["item_nbr"].unique())
        else:
            # CHỈ hiển thị items của store đã chọn
            filtered_by_store = data[data[store_column] == selected_store]
            available_items = sorted(filtered_by_store["item_nbr"].unique())
        # Selectbox với ĐÚNG available_items

        item_options = ["All Products"] + available_items
        selected_item = st.selectbox("Select Product", options=item_options)
        
        # ===== CATEGORY FILTER =====
        selected_categories = None
        if "category" in data.columns:
            st.subheader("Product Categories")
            categories = sorted(data["category"].unique())
            selected_categories = st.multiselect(
                "Select Categories", categories, default=categories
            )
    
    # ===== APPLY ALL FILTERS =====
    mask = (data["date"].dt.date >= start_date) & (data["date"].dt.date <= end_date)
    
    # Store filter
    if selected_store != "All Stores" and store_column:
        mask &= data[store_column] == selected_store
    
    # Item filter
    if selected_item != "All Products":
        mask &= data["item_nbr"] == selected_item
    
    # Category filter
    if selected_categories:
        mask &= data["category"].isin(selected_categories)
    
    # Determine store name for display
    if selected_store == "All Stores":
        selected_store_name = "All Stores"
    elif store_column == "store_name":
        selected_store_name = selected_store
    else:
        selected_store_name = f"Store {selected_store}"

    # Save to session state (nếu cần cho functions khác)
    st.session_state.selected_store = selected_store
    st.session_state.selected_store_name = selected_store_name
    st.session_state.selected_item = selected_item
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    
    return data[mask]


def display_kpis(filtered_data):
    """Display KPI metrics in the dashboard"""

    st.header("Key Performance Indicators")

    # Calculate KPIs
    total_units = filtered_data["units"].sum()
    avg_daily_units = filtered_data.groupby("date")["units"].sum().mean()

    # Calculate period comparison if enough data
    if len(filtered_data["date"].unique()) >= 2:
        # Split the date range in half for comparison
        mid_date = (
            st.session_state.start_date
            + (st.session_state.end_date - st.session_state.start_date) / 2
        )

        period1_data = filtered_data[filtered_data["date"].dt.date <= mid_date]
        period2_data = filtered_data[filtered_data["date"].dt.date > mid_date]

        period1_sales = period1_data["units"].sum() if not period1_data.empty else 0
        period2_sales = period2_data["units"].sum() if not period2_data.empty else 0

        sales_change_pct = (
            ((period2_sales - period1_sales) / period1_sales * 100)
            if period1_sales > 0
            else 0
        )
    else:
        sales_change_pct = 0

    # Transaction count if available
    if "transactions" in filtered_data.columns:
        total_transactions = filtered_data["transactions"].sum()
        avg_transaction_value = (
            total_units / total_transactions if total_transactions > 0 else 0
        )
    else:
        total_transactions = filtered_data.shape[0]  # Use row count as proxy
        avg_transaction_value = (
            total_units / total_transactions if total_transactions > 0 else 0
        )

    # Display KPIs in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Units Sold",
            value=f"{total_units:,.0f}",
            delta=f"{sales_change_pct:.1f}%" if sales_change_pct != 0 else None,
        )

    with col2:
        st.metric(label="Avg Daily Units", value=f"{avg_daily_units:,.1f}")

    with col3:
        st.metric(label="Total Transactions", value=f"{total_transactions:,}")

    with col4:
        st.metric(label="Avg Units/Trans", value=f"{avg_transaction_value:,.2f}")


def display_sales_trends(filtered_data):
    """Display sales trends section with time series and day of week patterns"""

    st.header("Sales Trends")

    col1, col2 = st.columns(2)

    with col1:
        # Time series plot of sales
        fig = plot_sales_time_series(
            filtered_data,
            st.session_state.selected_store,
            st.session_state.selected_store_name,
        )
        st.pyplot(fig)

    with col2:
        # Weekly patterns
        if len(filtered_data["date"].unique()) >= 7:
            fig = plot_day_of_week_pattern(filtered_data)
            st.pyplot(fig)


def display_performance_breakdown(filtered_data):
    """Display performance breakdown section with top products and store comparisons"""

    st.header("Performance Breakdown")
    
    # Check what content will be displayed
    has_multiple_products = (
        "item_nbr" in filtered_data.columns
        and len(filtered_data["item_nbr"].unique()) > 1
    )
    has_multiple_stores = (
        ("store_name" in filtered_data.columns or "store_nbr" in filtered_data.columns)
        and (
            ("store_name" in filtered_data.columns and len(filtered_data["store_name"].unique()) > 1)
            or ("store_nbr" in filtered_data.columns and len(filtered_data["store_nbr"].unique()) > 1)
        )
    )
    
    # Nothing to display
    if not has_multiple_products and not has_multiple_stores:
        st.info("Select 'All Stores' or 'All Products' to view performance breakdown.")
        return

    # Use two columns when both products and stores are available
    if has_multiple_products and has_multiple_stores:
        col1, col2 = st.columns(2)
        
        # Top Products in left column
        with col1:
            st.subheader("Top Products")
            product_sales = (
                filtered_data.groupby("item_nbr")["units"]
                .sum()
                .sort_values(ascending=False)
            )
            top_products = product_sales.head(10)
            product_df = pd.DataFrame({
                "Product": top_products.index,
                "Sales": top_products.values,
            })
            product_df["Sales"] = product_df["Sales"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(product_df, use_container_width=True)
            fig = plot_product_comparison(filtered_data, "item_nbr")
            st.pyplot(fig)
        
        # Top Stores in right column
        with col2:
            st.subheader("Top Stores")
            if "store_name" in filtered_data.columns:
                store_identifier = "store_name"
            else:
                store_identifier = "store_nbr"
            store_sales = (
                filtered_data.groupby(store_identifier)["units"]
                .sum()
                .sort_values(ascending=False)
            )
            top_stores = store_sales.head(10)
            store_df = pd.DataFrame({
                "Store": top_stores.index,
                "Sales": top_stores.values,
            })
            store_df["Sales"] = store_df["Sales"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(store_df, use_container_width=True)
            fig = plot_store_comparison(filtered_data, store_identifier)
            st.pyplot(fig)
    
    # Only Products - use full width
    elif has_multiple_products:
        st.subheader("Top Products")
        product_sales = (
            filtered_data.groupby("item_nbr")["units"]
            .sum()
            .sort_values(ascending=False)
        )
        top_products = product_sales.head(10)
        product_df = pd.DataFrame({
            "Product": top_products.index,
            "Sales": top_products.values,
        })
        product_df["Sales"] = product_df["Sales"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(product_df, use_container_width=True)
        fig = plot_product_comparison(filtered_data, "item_nbr")
        st.pyplot(fig)
    
    # Only Stores - use full width
    elif has_multiple_stores:
        st.subheader("Top Stores")
        if "store_name" in filtered_data.columns:
            store_identifier = "store_name"
        else:
            store_identifier = "store_nbr"
        store_sales = (
            filtered_data.groupby(store_identifier)["units"]
            .sum()
            .sort_values(ascending=False)
        )
        top_stores = store_sales.head(10)
        store_df = pd.DataFrame({
            "Store": top_stores.index,
            "Sales": top_stores.values,
        })
        store_df["Sales"] = store_df["Sales"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(store_df, use_container_width=True)
        fig = plot_store_comparison(filtered_data, store_identifier)
        st.pyplot(fig)
