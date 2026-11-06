"""SQLite database creation and population for Instamart Customer Intelligence & Support Copilot.

Tables created:
1. customers (customer_id PK, city, signup_date, age, gender)
2. products (product_id PK, product_name, category, price_inr)
3. orders (order_id PK, customer_id FK, product_id FK, quantity, order_ts, status, discount_pct, order_amount, delivery_minutes)

Validates row counts post-insertion.
"""

from pathlib import Path
import sqlite3
import pandas as pd


CREATE_TABLES_SQL = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    city TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL
);

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    price_inr REAL NOT NULL
);

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_ts TEXT NOT NULL,
    status TEXT NOT NULL,
    discount_pct REAL NOT NULL,
    order_amount REAL NOT NULL,
    delivery_minutes REAL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

CREATE INDEX idx_orders_customer_id ON orders (customer_id);
CREATE INDEX idx_orders_product_id ON orders (product_id);
CREATE INDEX idx_orders_status ON orders (status);
CREATE INDEX idx_orders_ts ON orders (order_ts);
"""


def build_database(
    clean_customers_path: Path | str = "data/clean/customers.csv",
    clean_products_path: Path | str = "data/clean/products.csv",
    clean_orders_path: Path | str = "data/clean/orders.csv",
    db_path: Path | str = "data/instamart.db"
) -> dict:
    """Build SQLite database from cleaned CSV files and validate counts."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    df_customers = pd.read_csv(clean_customers_path)
    df_products = pd.read_csv(clean_products_path)
    df_orders = pd.read_csv(clean_orders_path)

    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        cur.executescript(CREATE_TABLES_SQL)
        conn.commit()

        # Insert cleaned data
        df_customers.to_sql("customers", conn, if_exists="append", index=False)
        df_products.to_sql("products", conn, if_exists="append", index=False)
        df_orders.to_sql("orders", conn, if_exists="append", index=False)
        conn.commit()

        # Validate row counts
        counts = {}
        for tbl in ["customers", "products", "orders"]:
            cur.execute(f"SELECT COUNT(*) FROM {tbl};")
            counts[tbl] = cur.fetchone()[0]

        if counts["customers"] != len(df_customers):
            raise ValueError(f"customers count mismatch: {counts['customers']} vs {len(df_customers)}")
        if counts["products"] != len(df_products):
            raise ValueError(f"products count mismatch: {counts['products']} vs {len(df_products)}")
        if counts["orders"] != len(df_orders):
            raise ValueError(f"orders count mismatch: {counts['orders']} vs {len(df_orders)}")

        return {
            "status": "success",
            "db_path": str(db_file),
            "table_counts": counts
        }
    finally:
        conn.close()


if __name__ == "__main__":
    res = build_database()
    print("Database built successfully:", res["table_counts"])
