import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
