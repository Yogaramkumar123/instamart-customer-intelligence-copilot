"""RFM (Recency, Frequency, Monetary) analysis for Instamart Customer Segmentation.

Uses only completed orders.
Snapshot moment: 2025-10-01 00:00:00.

Metrics:
- Recency: Whole days between snapshot and latest completed order.
- Frequency: Number of distinct completed orders.
- Monetary: Sum of completed order_amount.
"""

from pathlib import Path
import pandas as pd


SNAPSHOT_MOMENT = pd.Timestamp("2025-10-01 00:00:00")


def compute_rfm_table(
    clean_orders_path: Path | str = "data/clean/orders.csv",
    snapshot: pd.Timestamp = SNAPSHOT_MOMENT
) -> tuple[pd.DataFrame, dict]:
    """Compute customer-level RFM metrics from completed orders."""
    df_orders = pd.read_csv(clean_orders_path)
    df_orders["order_ts"] = pd.to_datetime(df_orders["order_ts"])

    # Filter completed orders only
    completed = df_orders[df_orders["status"] == "completed"].copy()

    # Aggregate by customer_id
    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    # Whole days between snapshot and latest order
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days

    # Order columns
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]].sort_values("customer_id").reset_index(drop=True)

    # Compute summary stats
    stats = {
        "customer_count": int(len(rfm)),
        "recency_median": float(rfm["recency"].median()),
        "recency_min": int(rfm["recency"].min()),
        "recency_max": int(rfm["recency"].max()),
        "frequency_mean": round(float(rfm["frequency"].mean()), 2),
        "frequency_max": int(rfm["frequency"].max()),
        "monetary_mean": round(float(rfm["monetary"].mean()), 2),
        "monetary_max": round(float(rfm["monetary"].max()), 2),
    }

    return rfm, stats


if __name__ == "__main__":
    rfm_df, rfm_stats = compute_rfm_table()
    print("RFM Table Head:")
    print(rfm_df.head())
    print("\nRFM Statistics:")
    for k, v in rfm_stats.items():
        print(f"  {k}: {v}")
