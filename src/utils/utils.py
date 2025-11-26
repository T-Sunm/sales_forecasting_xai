import pickle

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

    # Step 1: Get non-zero items
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
