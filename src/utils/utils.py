import pickle

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from datetime import datetime


def get_value_column(df, possible_columns=None):
    """
    Auto-detect the target value column (sales or units).

    Parameters:
        df: DataFrame to search
        possible_columns: List of possible column names. Default is ['sales', 'units']

    Returns:
        str: Name of the found column

    Raises:
        ValueError: If no matching column is found
    """
    if possible_columns is None:
        possible_columns = ["sales", "units"]

    for col in possible_columns:
        if col in df.columns:
            return col

    raise ValueError(f"DataFrame must contain one of: {possible_columns}")


def get_nonzero_items(
    df, store_col="store_nbr", item_col="item_nbr", value_col="units"
):
    """
    Tìm các cặp (Store, Item) thực sự có bán hàng (tổng doanh số > 0).

    Parameters:
        df: DataFrame chứa dữ liệu bán hàng
        store_col: Tên cột store (mặc định 'store_nbr', có thể là 'store_id')
        item_col: Tên cột item (mặc định 'item_nbr', có thể là 'item_id')
        value_col: Tên cột giá trị (mặc định 'units', có thể là 'sales')

    Returns:
        DataFrame với 2 cột [store_col, item_col] chứa các cặp có doanh số
    """
    # Auto-detect column names if not exist
    if store_col not in df.columns:
        store_col = "store_id" if "store_id" in df.columns else "store_nbr"
    if item_col not in df.columns:
        item_col = "item_id" if "item_id" in df.columns else "item_nbr"
    if value_col not in df.columns:
        value_col = "sales" if "sales" in df.columns else "units"

    # Get unique store IDs
    store_ids = sorted(df[store_col].unique())

    nonzero_res = {}
    for store_id in store_ids:
        store_df = df[df[store_col] == store_id]
        # Calculate total sales per item
        item_totals = store_df.groupby(item_col)[value_col].sum().reset_index()
        # Keep only items with sales > 0
        nonzero_items = item_totals[item_totals[value_col] > 0]
        nonzero_res[store_id] = nonzero_items[item_col].values.tolist()

    # Convert to list of tuples
    retailed_items = []
    for store_id in store_ids:
        items = nonzero_res[store_id]
        pairs = list(zip([store_id] * len(items), items))
        retailed_items.extend(pairs)

    return pd.DataFrame(retailed_items, columns=[store_col, item_col])


def visualize_product_availability(
    retailed_items,
    store_col="store_nbr",
    item_col="item_nbr",
    figsize=(12, 15),
    title=None,
):
    """
    Hiển thị heatmap cho thấy sản phẩm nào được bán ở cửa hàng nào.

    Parameters:
        retailed_items: DataFrame từ get_nonzero_items()
        store_col: Tên cột store
        item_col: Tên cột item
        figsize: Kích thước figure
        title: Tiêu đề custom (nếu None sẽ dùng tiêu đề mặc định)
    """
    # Add marker column
    df = retailed_items.copy()
    df["available"] = 1

    # Create pivot table
    pivot = df.pivot_table(
        index=item_col, columns=store_col, values="available", aggfunc=np.sum
    )

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(pivot, cmap="RdYlGn", cbar_kws={"label": "Available"})
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)

    if title is None:
        title = "Product Availability Matrix\n(Red = Available, White = Not Available)"
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.xlabel(f'{store_col.replace("_", " ").title()}', fontsize=12)
    plt.ylabel(f'{item_col.replace("_", " ").title()}', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    total_stores = pivot.shape[1]
    total_items = pivot.shape[0]
    coverage_pct = (pivot.notna().sum().sum() / (total_stores * total_items)) * 100

    print(f"\n{'='*60}")
    print(f"SUMMARY: Product Distribution Analysis")
    print(f"{'='*60}")
    print(f"Total Stores: {total_stores}")
    print(f"Total Items: {total_items}")
    print(f"Total Possible Combinations: {total_stores * total_items:,}")
    print(f"Actual Products Sold: {len(retailed_items):,}")
    print(f"Coverage: {coverage_pct:.1f}%")
    print(f"{'='*60}\n")


def show_top_products(retailed_items, item_col="item_nbr", top_n=10):
    """
    Hiển thị top N sản phẩm được bán ở nhiều cửa hàng nhất.

    Parameters:
        retailed_items: DataFrame từ get_nonzero_items()
        item_col: Tên cột item
        top_n: Số lượng sản phẩm top muốn hiển thị

    Returns:
        DataFrame chứa top products
    """
    df = retailed_items.copy()
    df["count"] = 1

    # Count stores per item
    top_products = (
        df.groupby(item_col)["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "num_stores"})
        .sort_values("num_stores", ascending=False)
        .head(top_n)
    )

    print(f"\n{'='*60}")
    print(f"TOP {top_n} MOST WIDELY SOLD PRODUCTS")
    print(f"{'='*60}")
    print(top_products.to_string(index=False))
    print(f"{'='*60}\n")

    return top_products


def analyze_product_distribution(
    df,
    store_col=None,
    item_col=None,
    value_col=None,
    show_heatmap=True,
    show_top=True,
    top_n=10,
    figsize=(12, 15),
):
    """
    Phân tích toàn diện về phân bố sản phẩm giữa các cửa hàng.

    Parameters:
        df: DataFrame gốc chứa dữ liệu bán hàng
        store_col: Tên cột store (auto-detect nếu None)
        item_col: Tên cột item (auto-detect nếu None)
        value_col: Tên cột giá trị (auto-detect nếu None)
        show_heatmap: Có hiển thị heatmap không
        show_top: Có hiển thị top products không
        top_n: Số lượng top products
        figsize: Kích thước heatmap

    Returns:
        retailed_items: DataFrame các cặp (Store, Item) có bán hàng
    """
    print("\n🔍 ANALYZING PRODUCT DISTRIBUTION...\n")

    # Step 1: Get non-zero itemszzz
    retailed_items = get_nonzero_items(
        df,
        store_col=store_col or "store_nbr",
        item_col=item_col or "item_nbr",
        value_col=value_col or "units",
    )

    # Detect actual column names used
    actual_store_col = retailed_items.columns[0]
    actual_item_col = retailed_items.columns[1]

    print(f"✅ Found {len(retailed_items)} valid (Store, Item) pairs with sales > 0\n")

    # Step 2: Show top products
    if show_top:
        show_top_products(retailed_items, item_col=actual_item_col, top_n=top_n)

    # Step 3: Visualize heatmap
    if show_heatmap:
        visualize_product_availability(
            retailed_items,
            store_col=actual_store_col,
            item_col=actual_item_col,
            figsize=figsize,
        )

    return retailed_items


def fill_missing_values(df, value_col=None):
    """
    Fill NaN values in the target column with the mean of non-NaN values.

    Parameters:
        df: DataFrame containing the data
        value_col: (Optional) Name of the target column. If None, auto-detects 'sales' or 'units'

    Returns:
        DataFrame with filled values
    """
    df_filled = df.copy()

    # Auto-detect column if not specified
    if value_col is None:
        value_col = get_value_column(df_filled)

    df_filled[value_col] = df_filled[value_col].fillna(df_filled[value_col].mean())
    return df_filled


def fill_missing_values_grouped(df, value_col=None):
    df_filled = df.copy()

    if value_col is None:
        value_col = get_value_column(df_filled)

    store_col = "store_id" if "store_id" in df_filled.columns else "store_nbr"
    item_col = "item_id" if "item_id" in df_filled.columns else "item_nbr"

    # Fill NaN với mean của từng nhóm (store-item)
    df_filled[value_col] = df_filled.groupby([store_col, item_col])[
        value_col
    ].transform(lambda x: x.fillna(x.mean()))

    return df_filled


def correct_outliers(df, factor=3, value_col=None):
    """
    Identify and correct outliers in the target column by replacing them with the mean.
    Uses z-score method for outlier detection.

    Parameters:
        df: DataFrame containing the data
        factor: Z-score threshold (default=3, meaning 3 standard deviations)
        value_col: (Optional) Name of the target column. If None, auto-detects 'sales' or 'units'

    Returns:
        DataFrame with corrected outliers
    """
    df_corrected = df.copy()

    # Auto-detect column if not specified
    if value_col is None:
        value_col = get_value_column(df_corrected)

    # Calculate z-scores
    mean_val = df_corrected[value_col].mean()
    std_val = df_corrected[value_col].std()
    z_scores = (df_corrected[value_col] - mean_val) / std_val

    # Identify outliers
    outlier_indices = np.abs(z_scores) > factor

    # Correct outliers by replacing with mean
    df_corrected.loc[outlier_indices, value_col] = mean_val

    return df_corrected


def correct_weather_outliers(df, cols, method="iqr", factor=3):
    """
    Xử lý outlier cho các cột thời tiết.

    Parameters:
        df: DataFrame weather
        cols: Danh sách các cột cần check (ví dụ: ['tmax', 'tmin', 'stnpressure'])
        method: 'iqr' hoặc 'zscore'
        factor: Ngưỡng (1.5 hoặc 3 cho IQR, 3 cho Z-score)
    """
    df_clean = df.copy()

    for col in cols:
        # Skip if column is not numeric
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue

        if method == "iqr":
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
        else:  # z-score
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower = mean - factor * std
            upper = mean + factor * std

        # Identify outliers
        mask = (df_clean[col] < lower) | (df_clean[col] > upper)

        if mask.sum() > 0:
            print(
                f"Column '{col}': Found {mask.sum()} outliers. Range allowed: [{lower:.2f}, {upper:.2f}]"
            )

            # Fill NaN and interpolate , best for time series data
            df_clean.loc[mask, col] = np.nan
            df_clean[col] = df_clean[col].interpolate(method="linear")

    return df_clean


def get_holidays_from_web(year, country_code=1):
    """Lấy danh sách holidays từ timeanddate.com."""
    try:
        url = f"https://www.timeanddate.com/calendar/custom.html?year={year}&country={country_code}&cols=3&df=1&hol=25"
        response = requests.get(url, timeout=10)
        dom = BeautifulSoup(response.content, "html.parser")
        trs = dom.select("table.cht.lpad tr")

        holidays = []
        for tr in trs:
            try:
                datestr = tr.select_one("td:nth-of-type(1)").text
                holiday_name = tr.select_one("td:nth-of-type(2)").text
                date = datetime.strptime(f"{year} {datestr}", "%Y %b %d")
                holidays.append({"date": date, "holiday": holiday_name})
            except:
                continue

        return pd.DataFrame(holidays)
    except Exception as e:
        print(f"Failed to fetch holidays for {year}: {e}")
        return pd.DataFrame(columns=["date", "holiday"])


def get_blackfriday_dates():
    """Tạo danh sách Black Friday (4 ngày từ Thứ 6 sau Lễ Tạ Ơn)."""
    dates = [
        # 2012
        "2012-11-23",
        "2012-11-24",
        "2012-11-25",
        "2012-11-26",
        # 2013
        "2013-11-29",
        "2013-11-30",
        "2013-12-01",
        "2013-12-02",
        # 2014
        "2014-11-28",
        "2014-11-29",
        "2014-11-30",
        "2014-12-01",
    ]
    return pd.to_datetime(dates)


def add_date_features(df, date_col="date"):
    """
    Thêm date-related features (đúng cho US seasons).
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Basic date features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["week_of_year"] = df[date_col].dt.isocalendar().week
    df["quarter"] = df[date_col].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # === SEASON (US Meteorological Seasons) ===
    def get_us_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:  # [9, 10, 11]
            return 3  # Fall

    df["season"] = df["month"].apply(get_us_season)

    # Season name
    season_map = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
    df["season_name"] = df["season"].map(season_map)
    season_dummies = pd.get_dummies(
        df["season_name"], prefix="season", drop_first=True
    ).astype(int)
    df = pd.concat([df, season_dummies], axis=1)
    # Drop both season_name and the original numeric season column
    df = df.drop(["season_name", "season"], axis=1)
    print(f"✓ Encoded season into {len(season_dummies.columns)} dummy columns")

    # Holidays
    years = df["year"].unique()
    all_holidays = []

    print("Fetching US holidays...")
    for year in sorted(years):
        holiday_df = get_holidays_from_web(year, country_code=1)  # 1 = USA
        if not holiday_df.empty:
            holiday_df["lower_window"] = 0
            holiday_df["upper_window"] = 1
            all_holidays.append(holiday_df)

    if all_holidays:
        holidays_concat = pd.concat(all_holidays, ignore_index=True)
        holiday_dates = pd.to_datetime(holidays_concat["date"]).dt.date
        df["is_holiday"] = df[date_col].dt.date.isin(holiday_dates).astype(int)
    else:
        df["is_holiday"] = 0

    # Black Friday
    blackfriday_dates = get_blackfriday_dates().date
    df["is_blackfriday"] = df[date_col].dt.date.isin(blackfriday_dates).astype(int)

    print(f"✓ Added date features with US seasons")
    return df


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from matplotlib.offsetbox import AnchoredText
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_lag_scatter(x, lag=1, ax=None, **kwargs):
    """
    Vẽ scatter plot giữa giá trị hiện tại và giá trị lag.
    """
    x_lagged = x.shift(lag)
    y_current = x

    if ax is None:
        fig, ax = plt.subplots()

    scatter_kws = dict(alpha=0.75, s=3)
    line_kws = dict(color="C3")

    ax = sns.regplot(
        x=x_lagged,
        y=y_current,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        lowess=True,
        ax=ax,
        **kwargs,
    )

    # Hiển thị correlation coefficient
    corr = y_current.corr(x_lagged)
    at = AnchoredText(
        f"ρ = {corr:.3f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)

    ax.set_title(f"Lag {lag}", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{x.name} (t-{lag})", fontsize=10)
    ax.set_ylabel(f"{x.name} (t)", fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def plot_lag_analysis(series, lags=12, figsize=None, **kwargs):
    """
    Vẽ lag scatter plots để xem mối quan hệ giữa giá trị hiện tại và các lags.

    Parameters:
        series: pandas Series cần phân tích
        lags: Số lượng lags cần vẽ (mặc định 12)
        figsize: Kích thước figure (tự động nếu None)
    """
    nrows = 2
    ncols = math.ceil(lags / 2)

    if figsize is None:
        figsize = (ncols * 3, nrows * 3)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for ax, k in zip(fig.get_axes(), range(nrows * ncols)):
        if k + 1 <= lags:
            plot_lag_scatter(series, lag=k + 1, ax=ax)
        else:
            ax.axis("off")

    plt.suptitle(
        f"Lag Scatter Plots: {series.name}", fontsize=14, fontweight="bold", y=1.00
    )
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(series, lags=40, figsize=(14, 10), alpha=0.05):
    """
    Vẽ ACF và PACF plots để xác định số lags cần thiết.

    Parameters:
        series: pandas Series cần phân tích
        lags: Số lượng lags tối đa (mặc định 40)
        figsize: Kích thước figure
        alpha: Significance level (mặc định 0.05 = 95% confidence)
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # ACF Plot
    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=alpha)
    axes[0].set_title(
        f"Autocorrelation Function (ACF): {series.name}", fontsize=13, fontweight="bold"
    )
    axes[0].set_xlabel("Lag", fontsize=11)
    axes[0].set_ylabel("Autocorrelation", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # PACF Plot
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=alpha, method="ywm")
    axes[1].set_title(
        f"Partial Autocorrelation Function (PACF): {series.name}",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("Lag", fontsize=11)
    axes[1].set_ylabel("Partial Autocorrelation", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_autocorrelation(series, max_lags=40, scatter_lags=12):
    """
    Phân tích đầy đủ autocorrelation: Lag scatter + ACF/PACF.

    Parameters:
        series: pandas Series cần phân tích
        max_lags: Số lags tối đa cho ACF/PACF
        scatter_lags: Số lags cho lag scatter plots
    """
    print("=" * 70)
    print(f"AUTOCORRELATION ANALYSIS: {series.name}")
    print("=" * 70)
    print(f"Series length: {len(series)}")
    print(f"Non-null values: {series.notna().sum()}")
    print(f"Date range: {series.index.min()} to {series.index.max()}")
    print("\n")

    # 1. Lag scatter plots
    print("--- Lag Scatter Plots ---")
    plot_lag_analysis(series, lags=scatter_lags)

    # 2. ACF & PACF
    print("\n--- ACF & PACF Plots ---")
    plot_acf_pacf(series, lags=max_lags)

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE:")
    print("=" * 70)
    print("ACF (Autocorrelation Function):")
    print("  - Bars outside blue area = significant correlation")
    print("  - Slow decay = strong trend/seasonality")
    print("  - Sharp cutoff after lag k = MA(k) process")
    print("\nPACF (Partial Autocorrelation Function):")
    print("  - Bars outside blue area = direct influence")
    print("  - Sharp cutoff after lag k = AR(k) process")
    print("  - Use significant lags to create lag features")
    print("=" * 70)


def compute_optimal_lags(
    df,
    target_col="units",
    max_lag=30,
    group_cols=["store_nbr", "item_nbr"],
    confidence_level=0.95,
):
    """
    Tính autocorrelation và tìm lags SIGNIFICANT dựa trên confidence interval.

    Parameters:
        df: DataFrame
        target_col: Cột target
        max_lag: Số lags tối đa
        group_cols: Cột group
        confidence_level: Mức tin cậy (0.95 = 95%)

    Returns:
        dict với significant lags
    """
    from statsmodels.tsa.stattools import acf, pacf
    from scipy import stats

    df = df.copy()
    df["_group_id"] = df[group_cols].astype(str).agg("_".join, axis=1)

    all_groups = df["_group_id"].unique()

    # Lưu significant lags của từng group
    all_significant_lags = []
    group_sample_sizes = []

    for group_id in all_groups:
        group_data = df[df["_group_id"] == group_id].sort_values("date")[target_col]

        if len(group_data) > max_lag + 10:  # Cần đủ data
            try:
                # Tính PACF (partial autocorrelation)
                pacf_values = pacf(group_data.dropna(), nlags=max_lag, method="ywm")

                # Tính confidence interval cho group này
                N = len(group_data.dropna())
                z_score = stats.norm.ppf((1 + confidence_level) / 2)  # 1.96 cho 95%
                ci = z_score / np.sqrt(N)

                # Tìm lags significant (vượt CI)
                significant = [
                    lag for lag in range(1, max_lag + 1) if abs(pacf_values[lag]) > ci
                ]

                all_significant_lags.extend(significant)
                group_sample_sizes.append(N)

            except:
                continue

    # Đếm tần suất xuất hiện của mỗi lag
    from collections import Counter

    lag_counts = Counter(all_significant_lags)

    # Chọn lags xuất hiện ở nhiều groups (ví dụ: >30% groups)
    min_groups = len(all_groups) * 0.3
    frequent_lags = [lag for lag, count in lag_counts.items() if count >= min_groups]
    frequent_lags_sorted = sorted(
        frequent_lags, key=lambda x: lag_counts[x], reverse=True
    )

    # Stats
    avg_sample_size = np.mean(group_sample_sizes) if group_sample_sizes else 0
    avg_ci = (
        stats.norm.ppf((1 + confidence_level) / 2) / np.sqrt(avg_sample_size)
        if avg_sample_size > 0
        else 0
    )

    print(f"✓ Analyzed {len(group_sample_sizes)}/{len(all_groups)} groups")
    print(f"✓ Average sample size: {avg_sample_size:.0f} observations")
    print(
        f"✓ Average CI threshold: ±{avg_ci:.3f} ({int(confidence_level*100)}% confidence)"
    )
    print(f"✓ Found {len(frequent_lags)} lags significant in ≥30% of groups")
    print(f"✓ Top 10 lags: {frequent_lags_sorted[:10]}")

    return {
        "significant_lags": frequent_lags_sorted,
        "lag_frequencies": dict(lag_counts),
        "avg_ci_threshold": avg_ci,
        "stats": {
            "total_groups": len(all_groups),
            "analyzed_groups": len(group_sample_sizes),
            "avg_sample_size": avg_sample_size,
        },
    }


def create_rolling_features(
    df,
    target_col="logunits",
    windows=[7, 14, 28],
    statistics=["mean", "min", "max", "std"],
    group_cols=["store_nbr", "item_nbr"],
):
    """Tạo rolling features cho time series."""
    df = df.copy()

    # Tạo group key
    group_key = (
        df[group_cols].astype(str).agg("_".join, axis=1)
        if len(group_cols) > 1
        else df[group_cols[0]]
    )

    # Map statistics to pandas methods
    stat_funcs = {
        "mean": lambda x: x.shift(1).rolling(window, min_periods=1).mean(),
        "min": lambda x: x.shift(1).rolling(window, min_periods=1).min(),
        "max": lambda x: x.shift(1).rolling(window, min_periods=1).max(),
        "std": lambda x: x.shift(1).rolling(window, min_periods=1).std(),
    }

    feature_count = 0
    for window in windows:
        for stat in statistics:
            if stat in stat_funcs:
                col_name = f"{target_col}_{stat}_{window}d"
                # Inject window vào lambda
                func = (
                    lambda x, w=window, s=stat: x.shift(1)
                    .rolling(w, min_periods=1)
                    .agg(s)
                )
                df[col_name] = df.groupby(group_key)[target_col].transform(func)
                feature_count += 1

    print(f"✓ Created {feature_count} rolling features for '{target_col}'")
    return df


def create_ewma_features(
    df,
    target_col="sales",
    alphas=(0.5, 0.75),
    windows=(7, 14, 28),
    group_cols=("store_nbr", "item_nbr"),
    shift_before=True,
):
    """
    Tạo Exponentially Weighted Moving Average (EWMA) features cho time series.

    - EWMA giúp nhấn mạnh các ngày gần hiện tại hơn.
    - Luôn shift(1) để tránh leakage.

    Parameters
    ----------
    df : DataFrame
    target_col : str
        Cột target, ví dụ 'sales' hoặc 'logunits'.
    alphas : iterable
        Các smoothing factors (0<alpha<=1). alpha càng lớn → càng nhạy.
    windows : iterable
        Các window reference dùng để đặt tên feature (không dùng trong tính toán ewm).
    group_cols : tuple/list
        Các cột group, mặc định ('store_nbr','item_nbr').
    shift_before : bool
        True → shift(1) rồi mới ewm (khuyến nghị).

    Returns
    -------
    DataFrame với các cột ewma mới.
    """
    df = df.copy()

    # Tạo group key
    if isinstance(group_cols, (list, tuple)) and len(group_cols) > 1:
        group_key = df[list(group_cols)].astype(str).agg("_".join, axis=1)
    else:
        group_key = df[group_cols[0]]

    feature_count = 0

    for alpha in alphas:
        alpha_str = str(alpha).replace(".", "")  # 0.5 -> '05', 0.75 -> '075'

        for window in windows:
            col_name = f"{target_col}_ewma_{window}d_a{alpha_str}"

            def _ewma(x, a=alpha):
                s = x.shift(1) if shift_before else x
                return s.ewm(alpha=a, adjust=False).mean()

            df[col_name] = df.groupby(group_key)[target_col].transform(_ewma)
            feature_count += 1

    print(f"✓ Created {feature_count} EWMA features for '{target_col}'")
    return df


def remove_multicollinear_features(df, threshold=0.95, exclude_cols=None):
    """
    Bỏ các features có correlation > threshold.

    Parameters:
        df: DataFrame
        threshold: Ngưỡng correlation
        exclude_cols: List cột không được check/xóa (ví dụ: ['units', 'logunits'])
    """
    if exclude_cols is None:
        exclude_cols = []

    # Tính correlation matrix (chỉ numeric features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Loại các cột cần bảo vệ
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    corr_matrix = df[numeric_cols].corr().abs()

    # Lấy upper triangle
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Tìm features có correlation > threshold
    to_drop = [
        column for column in upper_tri.columns if any(upper_tri[column] > threshold)
    ]

    print(f"Features có correlation > {threshold}:")
    for col in to_drop:
        high_corr_with = upper_tri[col][upper_tri[col] > threshold].index.tolist()
        print(f"  - {col}: correlation cao với {high_corr_with}")

    df_cleaned = df.drop(columns=to_drop)

    print(f"\n✓ Đã bỏ {len(to_drop)} features: {to_drop}")
    print(f"✓ Còn lại {df_cleaned.shape[1]} features")
    print(f"✓ Protected columns: {exclude_cols}")

    return df_cleaned, to_drop


def get_sample_stores(df: pd.DataFrame, store_id: int = 1) -> pd.DataFrame:
    """
    Get the sample data for a specific store.
    Supports both 'store_id' and 'store_nbr' column names.

    Parameters:
        df: DataFrame containing store data
        store_id: ID of the store to filter

    Returns:
        Filtered DataFrame for the specified store
    """
    # Auto-detect store column name
    store_col = None
    for col_name in ["store_id", "store_nbr"]:
        if col_name in df.columns:
            store_col = col_name
            break

    if store_col is None:
        raise ValueError("DataFrame must contain 'store_id' or 'store_nbr' column")

    grouped = df.groupby(store_col)
    sample_store = grouped.get_group(store_id)
    return sample_store


def save_data(df, file_path, file_format="feather"):
    """
    Save a DataFrame to a specified file format.

    Parameters:
        df (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The path where the file will be saved.
        file_format (str): The format in which to save the file.
                          Supported formats: 'feather', 'csv'. Default is 'feather'.

    Example:
        ```
        save_data(df, 'output_data.feather', file_format='feather')
        ```
    """
    if file_format.lower() == "feather":
        df.to_feather(file_path)
        print(f"DataFrame saved to {file_path} in Feather format.")
    elif file_format.lower() == "csv":
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path} in CSV format.")
    else:
        print(
            f"Error: Unsupported file format '{file_format}'. "
            f"Supported formats: 'feather', 'csv'."
        )


def flatten_prophet_predictions(predictions_dict):
    """
    Flatten Prophet predictions dictionary into a single DataFrame.

    Parameters:
        predictions_dict: Dictionary with store-item keys and prediction DataFrames as values

    Returns:
        Concatenated DataFrame with 'store_item' column added
    """
    all_dfs = []

    for store_item, df in predictions_dict.items():
        df = df.copy()
        df["store_item"] = store_item
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def load_model(file_path):
    """
    Load a machine learning model from a file.
    Supports both scikit-learn (pickle) and LightGBM models.

    Parameters:
        file_path: The file path from where the model will be loaded.

    Returns:
        The loaded model.
    """
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
            print(f"Sklearn model loaded from {file_path}")
    except (pickle.UnpicklingError, FileNotFoundError):
        # If loading as scikit-learn model fails, assume it is a LightGBM model
        model = lgbm.Booster(model_file=file_path)
        print(f"LightGBM (scikit-learn API) model loaded from {file_path}")

    return model


def weighted_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Weighted Absolute Percentage Error (WAPE).

    Args:
        y_true: Actual values (array-like)
        y_pred: Predicted values (array-like)

    Returns:
        WAPE value (percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
