import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


def plot_sales_forecast(
    historical_data, prediction_date, prediction_value, store_id=None
):
    """
    Plot historical sales with prediction point
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter for specific store if provided
    if store_id is not None and "store_nbr" in historical_data.columns:
        plot_data = historical_data[historical_data["store_nbr"] == store_id].copy()
    else:
        plot_data = historical_data.copy()

    # Group by date if multiple records per date
    if len(plot_data) > len(plot_data["date"].unique()):
        plot_data = plot_data.groupby("date")["units"].sum().reset_index()

    # Sort by date
    plot_data = plot_data.sort_values("date")

    # Plot historical data
    ax.plot(plot_data["date"], plot_data["units"], label="Historical Sales")

    # Add prediction point
    ax.scatter(
        prediction_date, prediction_value, color="red", s=100, label="Prediction"
    )

    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    if store_id is not None:
        ax.set_title(f"Sales Forecast for Store {store_id}")
    else:
        ax.set_title("Sales Forecast")
    ax.legend()
    fig.autofmt_xdate()

    return fig


def plot_sales_time_series(
    filtered_data, selected_store=None, selected_store_name=None
):
    """Generate time series plot of sales with moving average"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot data based on store selection
    if selected_store_name == "All Stores" and selected_store == "All Stores":
        # Group by date for the trend line
        sales_by_date = filtered_data.groupby("date")["units"].sum()
        ax.plot(sales_by_date.index, sales_by_date.values, "b-")

        # Add moving average
        if len(sales_by_date) > 7:
            sales_by_date_df = sales_by_date.reset_index()
            sales_by_date_df["MA7"] = sales_by_date_df["units"].rolling(window=7).mean()
            ax.plot(
                sales_by_date_df["date"],
                sales_by_date_df["MA7"],
                "r--",
                label="7-Day Moving Avg",
            )
            ax.legend()
    else:
        # Single store - show daily sales and trend
        sales_by_date = filtered_data.groupby("date")["units"].sum()
        ax.plot(sales_by_date.index, sales_by_date.values, "b-")

        # Add moving average if enough data
        if len(sales_by_date) > 7:
            sales_by_date_df = sales_by_date.reset_index()
            sales_by_date_df["MA7"] = sales_by_date_df["units"].rolling(window=7).mean()
            ax.plot(
                sales_by_date_df["date"],
                sales_by_date_df["MA7"],
                "r--",
                label="7-Day Moving Avg",
            )
            ax.legend()

    ax.set_xlabel("")
    ax.set_ylabel("Units Sold")

    if "store_name" in filtered_data.columns and selected_store_name != "All Stores":
        ax.set_title(f"Daily Units - {selected_store_name}")
    elif "store_nbr" in filtered_data.columns and selected_store != "All Stores":
        ax.set_title(f"Daily Units - Store {selected_store}")
    else:
        ax.set_title("Daily Units - All Stores")

    fig.autofmt_xdate()
    return fig


def plot_day_of_week_pattern(filtered_data):
    """Generate bar chart showing sales by day of week"""
    filtered_data = filtered_data.copy()
    fig, ax = plt.subplots(figsize=(6, 4))

    # Add day of week name
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    filtered_data["day_name"] = filtered_data["date"].dt.dayofweek.apply(
        lambda x: day_names[x]
    )

    # Group by day of week
    day_sales = filtered_data.groupby("day_name")["units"].mean().reindex(day_names, fill_value=0)

    # Calculate average line
    avg_daily = day_sales.mean()

    # Create bar chart with average line
    bars = ax.bar(day_sales.index, day_sales.values, color="skyblue")
    ax.axhline(y=avg_daily, color="red", linestyle="--", label="Daily Average")

    # Highlight best and worst days
    best_day = day_sales.idxmax()
    worst_day = day_sales.idxmin()

    for i, (day, sales) in enumerate(day_sales.items()):
        if day == best_day:
            bars[i].set_color("green")
        elif day == worst_day:
            bars[i].set_color("orange")

    ax.set_xlabel("")
    ax.set_ylabel("Average Units")
    ax.set_title("Units by Day of Week")
    plt.xticks(rotation=45)
    ax.legend()

    return fig




def plot_store_comparison(filtered_data, store_identifier="store"):
    """Generate horizontal bar chart for top stores by sales"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Group by store
    store_sales = (
        filtered_data.groupby(store_identifier)["units"]
        .sum()
        .sort_values(ascending=False)
    )

    # Take top 10 stores
    top_stores = store_sales.head(10)

    # Plot horizontal bar chart
    y_pos = np.arange(len(top_stores))
    ax.barh(y_pos, top_stores.values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_stores.index)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel("Units Sold")
    ax.set_title("Top 10 Stores by Units")
    
    # Format x-axis with comma separators
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    return fig


def plot_product_comparison(filtered_data, item_identifier="item_nbr"):
    """Generate horizontal bar chart for top products by sales"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Group by product
    product_sales = (
        filtered_data.groupby(item_identifier)["units"]
        .sum()
        .sort_values(ascending=False)
    )

    # Take top 10 products
    top_products = product_sales.head(10)

    # Plot horizontal bar chart
    y_pos = np.arange(len(top_products))
    ax.barh(y_pos, top_products.values, align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_products.index)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel("Units Sold")
    ax.set_title("Top 10 Products by Units")
    
    # Format x-axis with comma separators
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    return fig


def plot_sales_distribution(df, col="store_day_units"):
    """Generate histogram with KDE and summary statistics for store-day units"""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Create histogram with KDE
    sns.histplot(df[col], bins=30, kde=True, ax=ax)
    
    # Add vertical lines for key statistics
    median_v = df[col].median()
    mean_v = df[col].mean()

    ax.axvline(
        x=median_v, color="r", linestyle="--", label=f"Median: {median_v:.0f}"
    )
    ax.axvline(
        x=mean_v, color="g", linestyle="--", label=f"Mean: {mean_v:.1f}"
    )
    
    ax.set_xlabel("Units per store-day")
    ax.set_ylabel("Frequency")
    ax.set_title("Store-day Units Distribution")
    ax.legend()

    return fig


def plot_products_trend_comparison(filtered_data, selected_items):
    """Generate time series overlay for selected products"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette matching consistency
    colors = plt.cm.tab10.colors

    for i, item in enumerate(selected_items):
        item_data = filtered_data[filtered_data["item_nbr"] == item]
        sales_by_date = item_data.groupby("date")["units"].sum()
        
        # Use simple color cycling
        color = colors[i % len(colors)]
        
        ax.plot(sales_by_date.index, sales_by_date.values, label=f"Product {item}", color=color, linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.set_title("Sales Trend Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    return fig


def plot_market_share_pie(filtered_data, selected_items):
    """Generate pie chart showing market share of selected products vs others"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate sales for selected items
    selected_sales = {}
    total_selected_sales = 0
    
    for item in selected_items:
        sales = filtered_data[filtered_data["item_nbr"] == item]["units"].sum()
        selected_sales[f"Product {item}"] = sales
        total_selected_sales += sales
    
    total_store_sales = filtered_data["units"].sum()
    others_sales = total_store_sales - total_selected_sales
    
    # Prepare data for pie chart
    labels = list(selected_sales.keys())
    sizes = list(selected_sales.values())
    
    # Only add "Others" if it's significant and positive
    if others_sales > 0:
        labels.append("Others")
        sizes.append(others_sales)

    # Plot
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%',
        startangle=90,
        textprops=dict(color="black")
    )
    
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title("Market Share Distribution")

    return fig


def plot_growth_rate_comparison(filtered_data, selected_items):
    """Generate bar chart comparing growth rates (First Half vs Second Half of period)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine mid-point of the date range
    if filtered_data.empty:
        return fig
        
    min_date = filtered_data["date"].min()
    max_date = filtered_data["date"].max()
    mid_date = min_date + (max_date - min_date) / 2
    
    growth_rates = []
    items = []
    
    for item in selected_items:
        item_data = filtered_data[filtered_data["item_nbr"] == item]
        
        first_half = item_data[item_data["date"] <= mid_date]["units"].sum()
        second_half = item_data[item_data["date"] > mid_date]["units"].sum()
        
        if first_half > 0:
            growth = ((second_half - first_half) / first_half) * 100
        else:
            growth = 0 if second_half == 0 else 100 # Handle edge case
            
        growth_rates.append(growth)
        items.append(f"Product {item}")
        
    # Plot bars
    colors = ['green' if g >= 0 else 'red' for g in growth_rates]
    bars = ax.barh(items, growth_rates, color=colors)
    
    # Add value labels
    ax.bar_label(bars, fmt='%.1f%%', padding=3)
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("Growth Rate (%)")
    ax.set_title(f"Growth Rate Comparison ({min_date.date()} - {mid_date.date()} vs {mid_date.date()} - {max_date.date()})")
    
    return fig


def plot_seasonality_heatmap(filtered_data, selected_items):
    """Generate heatmap showing sales intensity by Day of Week for selected products"""
    # Filter for selected items only
    df = filtered_data[filtered_data["item_nbr"].isin(selected_items)].copy()
    
    if df.empty:
        fig, ax = plt.subplots()
        return fig

    # Add day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = df['date'].dt.day_name()
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)
    
    # Group by Product and Day
    pivot_table = df.pivot_table(
        values='units', 
        index='item_nbr', 
        columns='day_of_week', 
        aggfunc='mean',
        observed=False
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, len(selected_items) * 0.8 + 2))
    
    sns.heatmap(
        pivot_table, 
        cmap="YlGnBu", 
        annot=True, 
        fmt=".1f", 
        linewidths=.5, 
        ax=ax,
        cbar_kws={'label': 'Avg Daily Units'}
    )
    
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Product ID")
    ax.set_title("Seasonality Heatmap (Avg Sales by Day)")
    
    return fig


def plot_hist_zoom_tail(df, col, p=0.99, bins=40, kde=False):
    """
    Plot histogram focusing on the main distribution (up to percentile p),
    clipping the heavy tail.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    x = df[col].dropna()
    
    # Handle empty or single value case
    if len(x) == 0:
        return fig
        
    hi = x.quantile(p)
    
    # If p99 is 0 or very small (sparse data), show everything or just 0
    if hi == 0:
        hi = x.max()
        if hi == 0: # All zeros
             sns.histplot(x, bins=bins, kde=False, ax=ax)
             ax.set_title(f"{col} distribution")
             return fig

    tail_cnt = int((x > hi).sum())
    tail_pct = 100 * tail_cnt / len(x)

    # Draw main part (<= p99)
    sns.histplot(x[x <= hi], bins=bins, kde=kde, ax=ax)

    ax.set_xlim(0, hi)
    ax.set_title(f"{col} distribution (zoom <= p{int(p*100)})")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

    ax.text(
        0.99, 0.95,
        f"Clipped: {tail_cnt} points ({tail_pct:.2f}%) > p{int(p*100)}",
        transform=ax.transAxes,
        ha="right", va="top"
    )
    return fig
