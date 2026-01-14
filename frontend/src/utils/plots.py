import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

from statsmodels.tsa.stattools import adfuller


def plot_custom_correlation(
    df,
    feature_cols,
    target_cols=["units", "store_nbr", "item_nbr", "station_nbr"],
    figsize=(16, 12),
    title_suffix="",
):
    """
    Vẽ Heatmap tương quan cho một nhóm Feature bất kỳ với Target.

    Parameters:
        df: DataFrame tổng
        feature_cols: List các cột tính năng muốn kiểm tra (VD: ['RA', 'SN'] hoặc ['tmax', 'tmin'])
        target_cols: Các cột mục tiêu/ID luôn muốn giữ lại để so sánh
    """
    # 1. Gộp danh sách cột cần vẽ
    cols_of_interest = target_cols + feature_cols

    # Lọc những cột thực sự tồn tại
    existing_cols = [c for c in cols_of_interest if c in df.columns]

    # Lấy dữ liệu số
    df_corr = df[existing_cols].select_dtypes(include=[np.number])

    if df_corr.empty:
        print("No suitable numeric data found for correlation.")
        return

    # 2. Tính Correlation
    corr_matrix = df_corr.corr()

    # 3. Vẽ Heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title(
        f"Correlation Matrix: {title_suffix} vs. Sales", fontsize=16, fontweight="bold"
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 4. In ra Top tương quan với 'units'
    if "units" in corr_matrix.columns:
        print(f"\n=== Top Correlation with Sales ({title_suffix}) ===")
        # Chỉ in ra các biến Feature, bỏ qua các biến ID
        relevant_features = [c for c in feature_cols if c in corr_matrix.columns]
        print(corr_matrix["units"][relevant_features].sort_values(ascending=False))


def plot_top_items_by_units(df, top_n=20, figsize=(14, 7)):
    """
    Vẽ biểu đồ Top N sản phẩm có tổng lượng bán (units) cao nhất.
    """
    # Tính tổng units cho mỗi item_nbr
    item_sales = (
        df.groupby("item_nbr")["logunits"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    # Chuyển item_nbr sang string để trục X hiển thị đúng dạng category
    item_sales["item_nbr"] = item_sales["item_nbr"].astype(str)

    plt.figure(figsize=figsize)
    sns.barplot(
        data=item_sales,
        x="item_nbr",
        y="logunits",
        palette="viridis",
        order=item_sales["item_nbr"],
    )

    plt.title(
        f"Top {top_n} Items by Total Log Units Sold", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Item Number", fontsize=12)
    plt.ylabel("Total Log Units Sold", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    from matplotlib.ticker import FuncFormatter

    def format_func(value, tick_number):
        return f"{int(value):,}"  # Thêm dấu phẩy ngăn cách hàng nghìn

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

    plt.tight_layout()
    plt.show()


def get_col_name(df, possible_names):
    """Helper to find the first existing column name from a list."""
    for col in possible_names:
        if col in df.columns:
            return col
    return None


def plot_sales(df, store_id=1, item_id=1, value_col=None):
    """
    Plot sales/units and visualize missing values.

    Parameters:
        df: DataFrame containing data
        store_id: ID of the store to plot
        item_id: ID of the item to plot
        value_col: (Optional) Name of the target column. If None, auto-detects 'sales' or 'units'.
    """

    # 1. Identify store_id column (store_id or store_nbr)
    store_col = get_col_name(df, ["store_id", "store_nbr"])
    if not store_col:
        raise ValueError("DataFrame must contain 'store_id' or 'store_nbr'")

    # 2. Identify item_id column (item_id or item_nbr)
    item_col = get_col_name(df, ["item_id", "item_nbr"])
    if not item_col:
        raise ValueError("DataFrame must contain 'item_id' or 'item_nbr'")

    # 3. Identify target value column (sales or units) if not provided
    if value_col is None:
        value_col = get_col_name(df, ["sales", "units"])
        if not value_col:
            raise ValueError(
                "DataFrame must contain 'sales' or 'units' column, or specify `value_col`"
            )

    # Filter data
    df_2plot = df.query(f"({store_col}==@store_id) & ({item_col}==@item_id)").copy()

    if df_2plot.empty:
        print(f"No data found for Store {store_id}, Item {item_id}")
        return

    # 4. Handle Store Name & Item Name safely
    store_name = (
        df_2plot["store_name"].iloc[-1]
        if "store_name" in df_2plot.columns
        else f"ID {store_id}"
    )
    item_name = (
        df_2plot["item_name"].iloc[-1]
        if "item_name" in df_2plot.columns
        else f"ID {item_id}"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )  # Increased size slightly for better visibility

    # Plot sales/units using the detected column name
    df_2plot.plot(x="date", y=value_col, ax=ax, legend=False)

    # Replace NaN values with the mean of surrounding two points (Visualization only)
    nan_indices = df_2plot[df_2plot[value_col].isna()].index

    if len(nan_indices) >= 1:
        # Fill NaNs for plotting continuity
        df_2plot[value_col] = df_2plot[value_col].fillna(method="ffill")

        # Draw markers for NaN values
        nan_dates = df_2plot.loc[nan_indices, "date"]
        nan_sales = df_2plot.loc[nan_indices, value_col]

        # Plot red dots/markers where data was missing
        ax.scatter(
            nan_dates, nan_sales, color="red", s=20, label="Missing (Filled)", zorder=5
        )

    # Set plot labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col.capitalize())  # "Sales" or "Units"
    ax.set_title(f"Store: {store_name} - Item: {item_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_forecast_single(flat_df, store_item):
    """
    Plot actual vs predicted sales for one store-item combo from flattened predictions for Prophet.
    """
    df = flat_df[flat_df["store_item"] == store_item].copy()

    if df.empty:
        print(f"No data found for: {store_item}")
        return

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="ds", y="y", label="Actual", color="black")
    sns.lineplot(data=df, x="ds", y="yhat", label="Forecast", color="blue")

    # Check if uncertainty intervals exist
    if "yhat_lower" in df.columns and "yhat_upper" in df.columns:
        plt.fill_between(
            df["ds"],
            df["yhat_lower"],
            df["yhat_upper"],
            color="blue",
            alpha=0.2,
            label="Confidence Interval",
        )

    plt.title(f"Forecast vs Actual for {store_item}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sales_predictions(
    df_prediction, store_id=1, nrows=6, ncols=5, figsize=(20, 20)
):
    """
    Plots actual vs predicted sales for items in a given store.

    Parameters:
        df_prediction (DataFrame): Must include ['store_id'/'store_nbr', 'item_id'/'item_nbr', 'date', 'sales', 'prediction']
        store_id (int): Store to filter on
        nrows (int): Rows of subplots
        ncols (int): Columns of subplots
        figsize (tuple): Size of the full figure
    """

    # 1. Identify store_id and item_id columns
    store_col = get_col_name(df_prediction, ["store_id", "store_nbr"])
    item_col = get_col_name(df_prediction, ["item_id", "item_nbr"])

    if not store_col or not item_col:
        raise ValueError(
            "DataFrame must contain 'store_id'/'store_nbr' and 'item_id'/'item_nbr'"
        )

    # Filter by store
    df_sample = df_prediction[df_prediction[store_col] == store_id].copy()

    if df_sample.empty:
        print(f"No prediction data found for Store {store_id}")
        return

    # 2. Handle Store Name safely
    store_name = (
        df_sample["store_name"].iloc[-1]
        if "store_name" in df_sample.columns
        else f"ID {store_id}"
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    item_ids = sorted(df_sample[item_col].unique())

    for i, ax in enumerate(axes):
        if i >= len(item_ids):
            ax.axis("off")  # Hide unused subplots
            continue

        current_item_id = item_ids[i]
        df2plot = df_sample[df_sample[item_col] == current_item_id]

        # 3. Handle Item Name safely
        item_name = (
            df2plot["item_name"].iloc[-1]
            if "item_name" in df2plot.columns
            else f"ID {current_item_id}"
        )

        if df2plot.empty:
            ax.axis("off")
            continue

        # Plot actual and predicted sales
        # Ensure date is datetime for proper plotting
        if not np.issubdtype(df2plot["date"].dtype, np.datetime64):
            df2plot["date"] = pd.to_datetime(df2plot["date"])

        ax.plot(df2plot["date"], df2plot["sales"], label="Actual", color="blue")

        # Check if prediction column exists
        if "prediction" in df2plot.columns:
            ax.plot(
                df2plot["date"],
                df2plot["prediction"],
                label="Forecast",
                color="red",
                linestyle="--",
                marker=".",
                markersize=2,  # smaller marker for cleaner look
            )

        ax.set_title(f"Item: {item_name}")
        ax.set_xlabel("")
        ax.set_ylabel("Sales")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    # Only add legend to the first subplot
    # Check if axes[0] actually has content to avoid empty legend warning
    if len(item_ids) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the legend
    fig.suptitle(
        f"Sales Forecast vs Actual - Store {store_name}", fontsize=16, fontweight="bold"
    )
    plt.show()


def plot_units_for_item(df, item_nbr, store_nbr=None, figsize=(12, 6), show_total=True):
    """
    Vẽ tổng units bán được theo thời gian cho một item nhất định (có thể chỉ định thêm store).

    Parameters:
      df: DataFrame bán hàng
      item_nbr: Giá trị/tập giá trị item (int hoặc list)
      store_nbr: Giá trị/tập giá trị store (int hoặc list) hoặc None để lấy tất cả
      figsize: Kích thước hình
      show_total: Nếu True, in tổng units dưới chart
    """
    # Nếu truyền vào một số, chuyển thành list cho tổng quát
    if not isinstance(item_nbr, (list, tuple, np.ndarray)):
        item_nbr = [item_nbr]

    # Filter item
    df_item = df[df["item_nbr"].isin(item_nbr)].copy()

    # Optional: Filter tiếp theo store nếu có
    if store_nbr is not None:
        if not isinstance(store_nbr, (list, tuple, np.ndarray)):
            store_nbr = [store_nbr]
        df_item = df_item[df_item["store_nbr"].isin(store_nbr)]

    if df_item.empty:
        print("Không có dữ liệu cho item_nbr và store_nbr chỉ định!")
        return

    # Nếu nhiều dòng/date, tổng units theo ngày
    df_daily = df_item.groupby("date")["units"].sum().reset_index()

    plt.figure(figsize=figsize)
    plt.plot(
        df_daily["date"],
        df_daily["units"],
        marker="o",
        color="blue",
        label="Units Sold",
    )
    plt.title(
        f"Units Sold Over Time for Item(s) {item_nbr}"
        + (f" (Store(s) {store_nbr})" if store_nbr else ""),
        fontsize=14,
    )
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()

    if show_total:
        print(f"Tổng units bán ra cho item(s) {item_nbr}: {df_item['units'].sum():,}")


def plot_units_by_store(df, figsize=(14, 7)):
    """
    Vẽ biểu đồ tổng units bán được theo từng cửa hàng.
    """
    # Tính tổng units cho mỗi store
    store_sales = (
        df.groupby("store_nbr")["units"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Chuyển store_nbr sang string
    store_sales["store_nbr"] = store_sales["store_nbr"].astype(str)

    plt.figure(figsize=figsize)
    sns.barplot(
        data=store_sales,
        x="store_nbr",
        y="units",
        palette="coolwarm",
        order=store_sales["store_nbr"],
    )

    plt.title("Total Units Sold by Store", fontsize=16, fontweight="bold")
    plt.xlabel("Store Number", fontsize=12)
    plt.ylabel("Total Units Sold", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Format trục Y với dấu phẩy
    from matplotlib.ticker import FuncFormatter

    def format_func(value, tick_number):
        return f"{int(value):,}"

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

    plt.tight_layout()
    plt.show()

    # In top 5 và bottom 5 stores
    print("=== Top 5 Stores by Sales ===")
    print(store_sales.head(5))
    print("\n=== Bottom 5 Stores by Sales ===")
    print(store_sales.tail(5))


def plot_weather_trend(df, weather_col="tmax", figsize=(14, 6)):
    """
    Vẽ xu hướng của một biến thời tiết theo thời gian (có thể nhiều trạm).

    Parameters:
        df: DataFrame thời tiết
        weather_col: Tên cột cần vẽ (tmax, tavg, preciptotal...)
        figsize: Kích thước hình
    """
    if weather_col not in df.columns:
        print(f"Column '{weather_col}' not found in DataFrame.")
        return

    plt.figure(figsize=figsize)

    # Nếu chỉ có 1 trạm hoặc đã aggregate
    df_sorted = df.sort_values("date")
    plt.plot(
        df_sorted["date"], df_sorted[weather_col], color="steelblue", linewidth=1.5
    )

    plt.title(f"Trend of {weather_col} over the dates", fontsize=14, fontweight="bold")
    plt.xlabel("dates", fontsize=11)
    plt.ylabel(weather_col, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)

    # Chỉ hiện legend nếu có nhiều trạm (tránh rối)
    if "station_nbr" in df.columns and df["station_nbr"].nunique() <= 10:
        plt.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_rolling_mean_stationarity(
    df, weather_cols, window=7, station_nbr=None, figsize=(14, 5)
):
    """
    Vẽ rolling mean + tự động chạy ADF test để kiểm tra stationarity.
    """
    # Filter data
    df_plot = df[df["station_nbr"] == station_nbr].copy() if station_nbr else df.copy()
    df_plot = df_plot.sort_values("date")

    # Lọc cột hợp lệ
    valid_cols = [
        c
        for c in weather_cols
        if c in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[c])
    ]

    if not valid_cols:
        print("No valid columns found.")
        return

    # Tính rolling
    for col in valid_cols:
        df_plot[f"{col}_rm"] = df_plot[col].rolling(window, min_periods=1).mean()
        df_plot[f"{col}_rs"] = df_plot[col].rolling(window, min_periods=1).std()

    # Plot
    fig, axes = plt.subplots(
        len(valid_cols), 1, figsize=(figsize[0], figsize[1] * len(valid_cols))
    )
    axes = [axes] if len(valid_cols) == 1 else axes

    for idx, col in enumerate(valid_cols):
        ax = axes[idx]
        ax.plot(
            df_plot["date"],
            df_plot[col],
            color="lightblue",
            linewidth=1,
            alpha=0.5,
            label="Raw",
        )
        ax.plot(
            df_plot["date"],
            df_plot[f"{col}_rm"],
            color="darkblue",
            linewidth=2,
            label=f"Rolling Mean ({window}d)",
        )
        ax.fill_between(
            df_plot["date"],
            df_plot[f"{col}_rm"] - df_plot[f"{col}_rs"],
            df_plot[f"{col}_rm"] + df_plot[f"{col}_rs"],
            color="blue",
            alpha=0.1,
            label="±1 Std",
        )

        ax.set_title(f"{col.upper()}", fontsize=11, fontweight="bold")
        ax.set_ylabel(col, fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(
        f"Stationarity Check: Rolling Mean Test", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    # ADF Test
    print("\n" + "=" * 70)
    print("AUGMENTED DICKEY-FULLER (ADF) TEST RESULTS")
    print("=" * 70)

    for col in valid_cols:
        series = df_plot[col].dropna()
        result = adfuller(series)
        status = "✅ STATIONARY" if result[1] < 0.05 else "❌ NON-STATIONARY"

        print(f"\n{col.upper()}: {status} (p-value = {result[1]:.4f})")
        print(f"  Test Stat: {result[0]:.4f} | Critical (5%): {result[4]['5%']:.4f}")


def plot_weather_boxplots(df, cols, ncols=3, figsize_factor=(5, 4)):
    """
    Vẽ hàng loạt Box Plots cho các cột thời tiết để kiểm tra Outliers.

    Parameters:
        df (DataFrame): DataFrame chứa dữ liệu thời tiết
        cols (list): Danh sách tên các cột cần vẽ (ví dụ: ['tmax', 'tmin', ...])
        ncols (int): Số lượng biểu đồ trên một hàng (mặc định 3)
        figsize_factor (tuple): Kích thước (rộng, cao) cho mỗi subplot
    """
    # Lọc ra chỉ các cột numeric có trong DataFrame
    valid_cols = [
        c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not valid_cols:
        print("No valid numeric columns found to plot.")
        return

    n_plots = len(valid_cols)
    nrows = math.ceil(n_plots / ncols)

    # Tự động tính kích thước tổng thể
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * figsize_factor[0], nrows * figsize_factor[1])
    )
    axes = axes.flatten()  # Làm phẳng mảng axes để dễ loop

    for i, col in enumerate(valid_cols):
        ax = axes[i]

        # Vẽ Box Plot
        sns.boxplot(
            x=df[col],
            ax=ax,
            color="skyblue",
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 3},
        )

        # Trang trí
        ax.set_title(f"{col}", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Ẩn các ô thừa
    for i in range(n_plots, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        "Weather Variables - Outlier Detection (Box Plots)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
