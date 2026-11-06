"""Audit raw datasets for Instamart Customer Intelligence & Support Copilot.

Loads:
- data/raw/customers.csv
- data/raw/products.csv
- data/raw/orders.csv
- data/raw/tickets.csv (using keep_default_na=False)

Reports shapes, data types, column missingness, and specific audit anomalies.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np


def audit_table_summary(df: pd.DataFrame, table_name: str) -> dict:
    """Compute basic structural audit for a dataframe."""
    missing_counts = df.isna().sum().to_dict()
    missing_pcts = ((df.isna().sum() / len(df)) * 100.0).round(4).to_dict()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    return {
        "table_name": table_name,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
    }


def run_audit(data_dir: Path | str = "data/raw", output_dir: Path | str | None = "outputs/part_a") -> dict:
    """Run full raw data audit against all raw files."""
    data_path = Path(data_dir)
    
    customers_path = data_path / "customers.csv"
    products_path = data_path / "products.csv"
    orders_path = data_path / "orders.csv"
    tickets_path = data_path / "tickets.csv"

    if not customers_path.exists():
        raise FileNotFoundError(f"Missing required file: {customers_path}")
    if not products_path.exists():
        raise FileNotFoundError(f"Missing required file: {products_path}")
    if not orders_path.exists():
        raise FileNotFoundError(f"Missing required file: {orders_path}")
    if not tickets_path.exists():
        raise FileNotFoundError(f"Missing required file: {tickets_path}")

    # Load raw tables
    df_customers = pd.read_csv(customers_path)
    df_products = pd.read_csv(products_path)
    df_orders = pd.read_csv(orders_path)
    df_tickets = pd.read_csv(tickets_path, keep_default_na=False)

    # General summaries
    summaries = {
        "customers": audit_table_summary(df_customers, "customers"),
        "products": audit_table_summary(df_products, "products"),
        "orders": audit_table_summary(df_orders, "orders"),
        "tickets": audit_table_summary(df_tickets, "tickets"),
    }

    # Orders specific metrics
    total_order_rows = int(len(df_orders))
    exact_duplicate_rows = int(df_orders.duplicated().sum())
    pending_rows = int((df_orders["status"] == "pending").sum())
    distinct_order_ids = int(df_orders["order_id"].nunique())
    
    numeric_amounts = pd.to_numeric(df_orders["order_amount"], errors="coerce")
    non_numeric_amounts = int(numeric_amounts.isna().sum())
    
    numeric_delivery = pd.to_numeric(df_orders["delivery_minutes"], errors="coerce")
    negative_delivery_times = int((numeric_delivery < 0).sum())
    blank_delivery_times = int(df_orders["delivery_minutes"].isna().sum())

    orders_audit = {
        "raw_order_rows": total_order_rows,
        "exact_duplicate_rows": exact_duplicate_rows,
        "pending_rows": pending_rows,
        "distinct_order_ids": distinct_order_ids,
        "non_numeric_amounts": non_numeric_amounts,
        "negative_delivery_times": negative_delivery_times,
        "blank_delivery_times": blank_delivery_times,
    }

    # Customers specific metrics
    blank_ages = int(df_customers["age"].isna().sum())
    numeric_ages = pd.to_numeric(df_customers["age"], errors="coerce")
    ages_outside_range = int(((numeric_ages < 10) | (numeric_ages > 100)).sum())
    distinct_raw_cities = int(df_customers["city"].nunique())
    
    raw_genders = df_customers["gender"].dropna()
    distinct_nonblank_genders = int(raw_genders[raw_genders.astype(str).str.strip() != ""].nunique())
    blank_genders = int(df_customers["gender"].isna().sum() + (df_customers["gender"].astype(str).str.strip() == "").sum())

    customers_audit = {
        "blank_ages": blank_ages,
        "ages_outside_10_100": ages_outside_range,
        "distinct_raw_city_spellings": distinct_raw_cities,
        "distinct_nonblank_raw_gender_spellings": distinct_nonblank_genders,
        "blank_genders": blank_genders,
    }

    results = {
        "summaries": summaries,
        "orders_audit": orders_audit,
        "customers_audit": customers_audit,
    }

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "audit_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    audit_data = run_audit()
    print("--- ORDERS AUDIT ---")
    for k, v in audit_data["orders_audit"].items():
        print(f"{k}: {v}")
    print("\n--- CUSTOMERS AUDIT ---")
    for k, v in audit_data["customers_audit"].items():
        print(f"{k}: {v}")
