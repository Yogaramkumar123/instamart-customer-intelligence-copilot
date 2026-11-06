"""Clean orders dataset for Instamart Customer Intelligence & Support Copilot.

Execution Sequence:
Step 1: Convert order_ts to datetime.
Step 2: Remove exact duplicate rows.
Step 3: For each order_id, retain only the record with the latest order_ts.
Step 4: Convert order_amount to numeric, turning unparseable entries into missing values,
        then fill each missing amount from product price:
        order_amount = round(price_inr * quantity * (1 - discount_pct / 100), 2).
Step 5: Convert negative delivery_minutes to missing values.
Step 6: For completed orders, fill missing delivery_minutes using the median delivery
        time of the corresponding product category among completed orders (groupby + transform).
Step 7: For cancelled orders, leave delivery_minutes missing.
Export to: data/clean/orders.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def clean_orders(
    raw_orders_path: Path | str = "data/raw/orders.csv",
    products_path: Path | str = "data/clean/products.csv",
    clean_orders_path: Path | str = "data/clean/orders.csv"
) -> tuple[pd.DataFrame, dict]:
    """Clean orders dataframe following the exact assignment sequence."""
    df_raw = pd.read_csv(raw_orders_path)
    df_prod = pd.read_csv(products_path)
    report = {}

    report["raw_order_rows"] = int(len(df_raw))

    # Step 1: Convert order_ts to datetime
    df_raw["order_ts"] = pd.to_datetime(df_raw["order_ts"])

    # Step 2: Remove exact duplicate rows
    initial_dups = int(df_raw.duplicated().sum())
    report["exact_duplicates_removed"] = initial_dups
    df_dedup = df_raw.drop_duplicates()

    # Step 3: For each order_id, retain only the record with the latest order_ts
    rows_before_latest = len(df_dedup)
    df_latest = df_dedup.sort_values("order_ts").groupby("order_id", as_index=False).last()
    report["superseded_records_removed"] = int(rows_before_latest - len(df_latest))
    report["distinct_order_ids"] = int(len(df_latest))

    # Step 4: Convert order_amount to numeric. Recover missing amounts using product catalog.
    df_latest["order_amount"] = pd.to_numeric(df_latest["order_amount"], errors="coerce")
    missing_amt_before = int(df_latest["order_amount"].isna().sum())
    report["unparseable_amounts_identified"] = missing_amt_before

    # Merge with product catalog to fetch price_inr and category
    df_merged = df_latest.merge(
        df_prod[["product_id", "price_inr", "category"]],
        on="product_id",
        how="left"
    )

    # Impute missing order_amounts
    missing_amt_mask = df_merged["order_amount"].isna()
    calc_amounts = (
        df_merged.loc[missing_amt_mask, "price_inr"]
        * df_merged.loc[missing_amt_mask, "quantity"]
        * (1.0 - df_merged.loc[missing_amt_mask, "discount_pct"] / 100.0)
    ).round(2)
    df_merged.loc[missing_amt_mask, "order_amount"] = calc_amounts
    report["amounts_recalculated_count"] = int(missing_amt_mask.sum())
    report["missing_amounts_after_cleaning"] = int(df_merged["order_amount"].isna().sum())

    # Step 5: Convert negative delivery_minutes to missing values
    df_merged["delivery_minutes"] = pd.to_numeric(df_merged["delivery_minutes"], errors="coerce")
    neg_mask = df_merged["delivery_minutes"] < 0
    report["negative_delivery_minutes_coerced"] = int(neg_mask.sum())
    df_merged.loc[neg_mask, "delivery_minutes"] = np.nan

    # Step 6: For completed orders, fill missing delivery_minutes with median of product category among completed orders
    completed_mask = df_merged["status"] == "completed"
    missing_deliv_completed_before = int(df_merged.loc[completed_mask, "delivery_minutes"].isna().sum())
    report["missing_delivery_completed_before_imputation"] = missing_deliv_completed_before

    category_medians = df_merged[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    df_merged.loc[completed_mask, "delivery_minutes"] = df_merged.loc[completed_mask, "delivery_minutes"].fillna(category_medians)

    # Step 7: For cancelled orders, leave delivery_minutes missing (never impute)
    cancelled_mask = df_merged["status"] == "cancelled"
    report["missing_delivery_completed_final"] = int(df_merged.loc[completed_mask, "delivery_minutes"].isna().sum())
    report["missing_delivery_cancelled_final"] = int(df_merged.loc[cancelled_mask, "delivery_minutes"].isna().sum())

    # Format order_ts as string in YYYY-MM-DD HH:MM:SS format
    df_merged["order_ts"] = df_merged["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Select original columns for clean orders table
    clean_cols = [
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "order_ts",
        "status",
        "discount_pct",
        "order_amount",
        "delivery_minutes"
    ]
    df_clean = df_merged[clean_cols].copy()

    # Final summary metrics
    report["total_orders"] = int(len(df_clean))
    report["completed_orders"] = int((df_clean["status"] == "completed").sum())
    report["cancelled_orders"] = int((df_clean["status"] == "cancelled").sum())
    report["pending_orders"] = int((df_clean["status"] == "pending").sum())
    completed_rev = float(df_clean.loc[df_clean["status"] == "completed", "order_amount"].sum())
    report["completed_revenue"] = round(completed_rev, 2)

    # Export to clean directory
    out_file = Path(clean_orders_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_file, index=False)

    return df_clean, report


if __name__ == "__main__":
    clean_df, rep = clean_orders()
    print("Total orders:", rep["total_orders"])
    print("Completed orders:", rep["completed_orders"])
    print("Cancelled orders:", rep["cancelled_orders"])
    print("Pending orders:", rep["pending_orders"])
    print("Missing amounts after:", rep["missing_amounts_after_cleaning"])
    print("Missing delivery (completed):", rep["missing_delivery_completed_final"])
    print("Missing delivery (cancelled):", rep["missing_delivery_cancelled_final"])
    print(f"Completed revenue: INR {rep['completed_revenue']:,.2f}")
