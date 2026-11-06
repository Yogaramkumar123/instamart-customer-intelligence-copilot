"""Master execution script for Part A of Instamart Copilot.

Orchestrates:
1. Raw Data Generation check.
2. Raw Data Audit.
3. Customer Cleaning.
4. Product Cleaning.
5. Order Cleaning.
6. SQLite Database Creation & Population.
7. Seven SQL Queries Execution.
8. Pandas Exploratory & Descriptive Statistical Analysis.
9. Automated Validation Suite.
"""

from pathlib import Path
import sys
import subprocess

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.part_a.audit import run_audit
from src.part_a.clean_customers import clean_customers
from src.part_a.clean_products import clean_products
from src.part_a.clean_orders import clean_orders
from src.part_a.database import build_database
from src.part_a.sql_analysis import run_sql_analysis
from src.part_a.pandas_analysis import run_pandas_analysis
from src.part_a.validate import validate_part_a


def run_pipeline():
    print("=" * 60)
    print("INSTAMART COPILOT — PART A PIPELINE EXECUTION")
    print("=" * 60)

    # 1. Ensure raw datasets exist
    raw_dir = REPO_ROOT / "data" / "raw"
    if not (raw_dir / "customers.csv").exists() or not (raw_dir / "orders.csv").exists():
        print("\n[Step 1] Generating raw synthetic datasets via generate_data.py...")
        subprocess.run([sys.executable, str(REPO_ROOT / "generate_data.py")], check=True)
    else:
        print("\n[Step 1] Raw synthetic datasets already present.")

    # 2. Raw data audit
    print("\n[Step 2] Auditing raw datasets...")
    audit_res = run_audit(
        data_dir=REPO_ROOT / "data" / "raw",
        output_dir=REPO_ROOT / "outputs" / "part_a"
    )
    print("  Raw orders audit:", audit_res["orders_audit"])
    print("  Raw customers audit:", audit_res["customers_audit"])

    # 3. Clean customers
    print("\n[Step 3] Cleaning customers dataset...")
    df_cust, cust_report = clean_customers(
        raw_path=REPO_ROOT / "data" / "raw" / "customers.csv",
        clean_path=REPO_ROOT / "data" / "clean" / "customers.csv"
    )
    print("  Cleaned customers count:", len(df_cust))
    print("  City distribution:", cust_report["city_counts"])
    print("  Gender distribution:", cust_report["gender_counts"])

    # 4. Clean products
    print("\n[Step 4] Cleaning and validating products catalog...")
    df_prod, prod_report = clean_products(
        raw_path=REPO_ROOT / "data" / "raw" / "products.csv",
        clean_path=REPO_ROOT / "data" / "clean" / "products.csv"
    )
    print("  Cleaned products count:", len(df_prod))
    print("  Categories:", prod_report["categories"])

    # 5. Clean orders
    print("\n[Step 5] Cleaning orders dataset...")
    df_orders, orders_report = clean_orders(
        raw_orders_path=REPO_ROOT / "data" / "raw" / "orders.csv",
        products_path=REPO_ROOT / "data" / "clean" / "products.csv",
        clean_orders_path=REPO_ROOT / "data" / "clean" / "orders.csv"
    )
    print("  Total clean orders:", orders_report["total_orders"])
    print("  Completed orders:", orders_report["completed_orders"])
    print("  Cancelled orders:", orders_report["cancelled_orders"])
    print(f"  Completed revenue: INR {orders_report['completed_revenue']:,.2f}")

    # 6. SQLite Database Creation
    print("\n[Step 6] Storing clean datasets into SQLite database (data/instamart.db)...")
    db_res = build_database(
        clean_customers_path=REPO_ROOT / "data" / "clean" / "customers.csv",
        clean_products_path=REPO_ROOT / "data" / "clean" / "products.csv",
        clean_orders_path=REPO_ROOT / "data" / "clean" / "orders.csv",
        db_path=REPO_ROOT / "data" / "instamart.db"
    )
    print("  Database tables inserted:", db_res["table_counts"])

    # 7. SQL Analysis
    print("\n[Step 7] Executing all seven SQL analytical queries...")
    sql_res = run_sql_analysis(
        db_path=REPO_ROOT / "data" / "instamart.db",
        sql_file_path=REPO_ROOT / "sql" / "queries.sql",
        output_dir=REPO_ROOT / "outputs" / "part_a"
    )
    print("  Q1 qualifying customers:", sql_res["Q1"]["row_count"])
    print("  Q2 revenue cells:", sql_res["Q2"]["row_count"], "Grand Total: INR", sql_res["Q2"]["grand_total"])
    print("  Q3 above-avg products:", sql_res["Q3"]["row_count"])
    print("  Q4 zero-completed customers:", sql_res["Q4"]["row_count"])
    print("  Q5 above-avg spend customers:", sql_res["Q5"]["row_count"])
    print("  Q6 rank tie differences:", sql_res["Q6"]["rank_row_number_diff_count"])
    print("  Q7 daily revenue rows:", sql_res["Q7"]["row_count"], "Final Cumulative: INR", sql_res["Q7"]["final_cumulative_revenue"])

    # 8. Pandas Analysis & Statistics
    print("\n[Step 8] Running Pandas cross-check and descriptive statistics...")
    pd_res = run_pandas_analysis(
        clean_orders_path=REPO_ROOT / "data" / "clean" / "orders.csv",
        clean_customers_path=REPO_ROOT / "data" / "clean" / "customers.csv",
        clean_products_path=REPO_ROOT / "data" / "clean" / "products.csv",
        output_dir=REPO_ROOT / "outputs" / "part_a"
    )
    print("  Pivot Table Grand Total: INR", pd_res["grand_total_pivot"])
    print("  Completed order_amount stats:", pd_res["descriptive_stats"])
    print("  Top correlation pair:", pd_res["largest_correlation_pair"])

    # 9. Automated Validation
    print("\n[Step 9] Running automated checkpoint validation...")
    validate_part_a(
        clean_dir=REPO_ROOT / "data" / "clean",
        db_path=REPO_ROOT / "data" / "instamart.db",
        sql_path=REPO_ROOT / "sql" / "queries.sql"
    )


if __name__ == "__main__":
    run_pipeline()
