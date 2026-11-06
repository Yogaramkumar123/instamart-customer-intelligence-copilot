"""Comprehensive automated validation for Part A data pipeline.

Verifies:
1. Required clean files and database exist.
2. Database tables exist and row counts match clean CSVs.
3. Customer checkpoints (city counts, gender counts, 0 missing ages, ages in 18-60, 6 cities).
4. Order checkpoints (total 12,309, completed 11,110, cancelled 1,199, pending 0,
   0 missing amounts, completed delivery times complete, cancelled delivery times NaN,
   completed revenue INR 2,998,668.80).
5. All seven SQL queries return reference numbers.
6. Pivot table cross-check and outlier statistics.

Prints exactly 'ALL CHECKPOINTS PASSED' upon complete success.
"""

from pathlib import Path
import sys
import sqlite3
import pandas as pd
import numpy as np

from src.part_a.sql_analysis import run_sql_analysis
from src.part_a.pandas_analysis import run_pandas_analysis


def assert_checkpoint(name: str, actual, expected, tol: float | None = None):
    """Assert actual equals expected, reporting clear diagnostic upon mismatch."""
    if tol is not None:
        if abs(actual - expected) > tol:
            print(f"FAILED CHECKPOINT: [{name}] Expected {expected} (+/- {tol}), got {actual}")
            raise AssertionError(f"Checkpoint [{name}] failed: Expected {expected}, got {actual}")
    else:
        if actual != expected:
            print(f"FAILED CHECKPOINT: [{name}] Expected {expected}, got {actual}")
            raise AssertionError(f"Checkpoint [{name}] failed: Expected {expected}, got {actual}")
    print(f"  [PASS] {name}: {actual}")


def validate_part_a(
    clean_dir: Path | str = "data/clean",
    db_path: Path | str = "data/instamart.db",
    sql_path: Path | str = "sql/queries.sql"
) -> bool:
    """Run full validation suite for Part A."""
    print("==================================================")
    print("STARTING PART A COMPREHENSIVE VALIDATION")
    print("==================================================")

    c_dir = Path(clean_dir)
    cust_file = c_dir / "customers.csv"
    prod_file = c_dir / "products.csv"
    orders_file = c_dir / "orders.csv"
    db_file = Path(db_path)
    sql_file = Path(sql_path)

    # 1. Required files exist
    print("\n--- Verifying File Existence ---")
    assert_checkpoint("Clean Customers CSV Exists", cust_file.exists(), True)
    assert_checkpoint("Clean Products CSV Exists", prod_file.exists(), True)
    assert_checkpoint("Clean Orders CSV Exists", orders_file.exists(), True)
    assert_checkpoint("SQLite Database Exists", db_file.exists(), True)
    assert_checkpoint("SQL Queries File Exists", sql_file.exists(), True)

    # Load cleaned dataframes
    df_cust = pd.read_csv(cust_file)
    df_prod = pd.read_csv(prod_file)
    df_orders = pd.read_csv(orders_file)

    # 2. Database tables and row counts
    print("\n--- Verifying Database Schema and Row Counts ---")
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        assert_checkpoint("Customers Table in DB", "customers" in tables, True)
        assert_checkpoint("Products Table in DB", "products" in tables, True)
        assert_checkpoint("Orders Table in DB", "orders" in tables, True)

        for tbl, df in [("customers", df_cust), ("products", df_prod), ("orders", df_orders)]:
            cur.execute(f"SELECT COUNT(*) FROM {tbl};")
            db_cnt = cur.fetchone()[0]
            assert_checkpoint(f"DB Row Count ({tbl})", db_cnt, len(df))
    finally:
        conn.close()

    # 3. Customer Cleaning Checkpoints
    print("\n--- Verifying Customer Checkpoints ---")
    assert_checkpoint("Customer Count", len(df_cust), 1200)
    assert_checkpoint("Distinct Cities Count", df_cust["city"].nunique(), 6)
    
    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }
    actual_cities = df_cust["city"].value_counts().to_dict()
    assert_checkpoint("Customer City Distribution", actual_cities, expected_cities)

    expected_genders = {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }
    actual_genders = df_cust["gender"].value_counts().to_dict()
    assert_checkpoint("Customer Gender Distribution", actual_genders, expected_genders)

    assert_checkpoint("Customer Missing Ages", int(df_cust["age"].isna().sum()), 0)
    assert_checkpoint("Customer Min Age >= 18", bool(df_cust["age"].min() >= 18), True)
    assert_checkpoint("Customer Max Age <= 60", bool(df_cust["age"].max() <= 60), True)

    # 4. Product Checkpoints
    print("\n--- Verifying Product Checkpoints ---")
    assert_checkpoint("Product Count", len(df_prod), 40)
    assert_checkpoint("Distinct Product IDs", df_prod["product_id"].nunique(), 40)
    assert_checkpoint("All Prices Positive", bool((df_prod["price_inr"] > 0).all()), True)

    # 5. Order Cleaning Checkpoints
    print("\n--- Verifying Order Checkpoints ---")
    assert_checkpoint("Total Orders Count", len(df_orders), 12309)
    assert_checkpoint("Distinct Order IDs Count", df_orders["order_id"].nunique(), 12309)

    completed_orders = df_orders[df_orders["status"] == "completed"]
    cancelled_orders = df_orders[df_orders["status"] == "cancelled"]
    pending_orders = df_orders[df_orders["status"] == "pending"]

    assert_checkpoint("Completed Orders Count", len(completed_orders), 11110)
    assert_checkpoint("Cancelled Orders Count", len(cancelled_orders), 1199)
    assert_checkpoint("Pending Orders Count", len(pending_orders), 0)

    assert_checkpoint("Missing Order Amounts", int(df_orders["order_amount"].isna().sum()), 0)
    assert_checkpoint(
        "Missing Delivery Times in Completed Orders",
        int(completed_orders["delivery_minutes"].isna().sum()),
        0
    )
    assert_checkpoint(
        "Missing Delivery Times in Cancelled Orders",
        int(cancelled_orders["delivery_minutes"].isna().sum()),
        1199
    )

    completed_rev = round(float(completed_orders["order_amount"].sum()), 2)
    assert_checkpoint("Completed Order Revenue (INR)", completed_rev, 2998668.80, tol=0.01)

    # 6. SQL Queries Checkpoints
    print("\n--- Verifying Seven SQL Queries ---")
    sql_results = run_sql_analysis(db_path=db_file, sql_file_path=sql_file)

    assert_checkpoint("Q1 Qualifying Customers Count", sql_results["Q1"]["row_count"], 30)
    assert_checkpoint("Q2 Cell Count", sql_results["Q2"]["row_count"], 30)
    assert_checkpoint("Q2 Grand Total (INR)", sql_results["Q2"]["grand_total"], 2998668.80, tol=0.01)
    assert_checkpoint(
        "Q2 Bengaluru Electronics Revenue (INR)",
        sql_results["Q2"]["bengaluru_electronics"],
        233909.45,
        tol=0.01
    )
    assert_checkpoint("Q3 Above-Average Products Count", sql_results["Q3"]["row_count"], 11)
    assert_checkpoint("Q4 Zero-Completed Customers Count", sql_results["Q4"]["row_count"], 25)
    assert_checkpoint("Q5 Above-Average Spend Customers Count", sql_results["Q5"]["row_count"], 550)
    assert_checkpoint("Q5 Average Spend (INR)", sql_results["Q5"]["average_spend"], 2552.06, tol=0.01)

    expected_q6_tops = {
        "Bengaluru": "C0082",
        "Chennai": "C0930",
        "Delhi": "C0291",
        "Hyderabad": "C0743",
        "Mumbai": "C0693",
        "Pune": "C0426",
    }
    assert_checkpoint("Q6 Top Customers per City", sql_results["Q6"]["top_by_city"], expected_q6_tops)
    assert_checkpoint("Q6 Ties (RANK != ROW_NUMBER)", sql_results["Q6"]["rank_row_number_diff_count"], 8)

    assert_checkpoint("Q7 Daily Rows Count", sql_results["Q7"]["row_count"], 183)
    assert_checkpoint("Q7 Final Cumulative Revenue (INR)", sql_results["Q7"]["final_cumulative_revenue"], 2998668.80, tol=0.01)

    # 7. Pandas Cross-check and Statistics
    print("\n--- Verifying Pandas Analysis & Statistics ---")
    pd_results = run_pandas_analysis(
        clean_orders_path=orders_file,
        clean_customers_path=cust_file,
        clean_products_path=prod_file
    )

    assert_checkpoint("Pivot Table Grand Total (INR)", pd_results["grand_total_pivot"], 2998668.80, tol=0.01)
    
    blr_stats = pd_results["city_metrics"]["Bengaluru"]
    assert_checkpoint("Bengaluru Sum Spend (INR)", blr_stats["sum_order_amount"], 523643.40, tol=0.01)
    assert_checkpoint("Bengaluru Mean Spend (INR)", blr_stats["mean_order_amount"], 276.47, tol=0.01)
    assert_checkpoint("Bengaluru Order Count", blr_stats["order_count"], 1894)

    mum_stats = pd_results["city_metrics"]["Mumbai"]
    assert_checkpoint("Mumbai Sum Spend (INR)", mum_stats["sum_order_amount"], 479058.50, tol=0.01)
    assert_checkpoint("Mumbai Mean Spend (INR)", mum_stats["mean_order_amount"], 297.00, tol=0.01)
    assert_checkpoint("Mumbai Order Count", mum_stats["order_count"], 1613)

    desc = pd_results["descriptive_stats"]
    assert_checkpoint("Completed Amount Mean (INR)", desc["mean"], 269.91, tol=0.01)
    assert_checkpoint("Completed Amount Median (INR)", desc["median"], 135.00, tol=0.01)
    assert_checkpoint("Completed Amount Skewness", desc["skewness"], 4.61, tol=0.05)
    assert_checkpoint("Completed Amount Q1 (INR)", desc["q1"], 64.80, tol=0.01)
    assert_checkpoint("Completed Amount Q3 (INR)", desc["q3"], 297.00, tol=0.01)
    assert_checkpoint("Completed Amount Upper Fence (INR)", desc["upper_fence"], 645.30, tol=0.01)
    assert_checkpoint("Lower Outliers Count", desc["lower_outliers"], 0)
    assert_checkpoint("Upper Outliers Count", desc["upper_outliers"], 897)

    top_corr = pd_results["largest_correlation_pair"]
    assert_checkpoint("Top Correlation Variables", set(top_corr["pair"]), {"order_amount", "discount_pct"})
    assert_checkpoint("Top Correlation Value", top_corr["correlation"], -0.095, tol=0.005)

    print("\n==================================================")
    print("ALL CHECKPOINTS PASSED")
    print("==================================================")
    return True


if __name__ == "__main__":
    try:
        success = validate_part_a()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
