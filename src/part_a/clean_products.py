"""Clean and validate products dataset for Instamart Customer Intelligence & Support Copilot.

Preserves:
- product_id
- product_name
- category
- price_inr

Validates:
- Unique product IDs
- Valid categories
- Positive prices (> 0)
- Export to data/clean/products.csv
"""

from pathlib import Path
import pandas as pd


VALID_CATEGORIES = {"Grocery", "Dairy", "Snacks", "Personal Care", "Electronics"}


def clean_products(
    raw_path: Path | str = "data/raw/products.csv",
    clean_path: Path | str = "data/clean/products.csv"
) -> tuple[pd.DataFrame, dict]:
    """Clean and validate products dataset."""
    df = pd.read_csv(raw_path)
    report = {}

    report["initial_rows"] = len(df)
    expected_cols = ["product_id", "product_name", "category", "price_inr"]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Products table missing expected columns: {expected_cols}")

    df = df[expected_cols].copy()

    # Validate unique product IDs
    if df["product_id"].nunique() != len(df):
        raise ValueError("Duplicate product_id found in products table.")
    report["unique_product_ids"] = len(df)

    # Validate categories
    found_categories = set(df["category"].unique())
    if not found_categories.issubset(VALID_CATEGORIES):
        invalid = found_categories - VALID_CATEGORIES
        raise ValueError(f"Found invalid categories: {invalid}")
    report["categories"] = sorted(list(found_categories))
    report["category_counts"] = df["category"].value_counts().to_dict()

    # Validate prices
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    if df["price_inr"].isna().any():
        raise ValueError("Found non-numeric prices in products table.")
    if (df["price_inr"] <= 0).any():
        raise ValueError("Found non-positive prices in products table.")
    report["min_price"] = float(df["price_inr"].min())
    report["max_price"] = float(df["price_inr"].max())

    # Export cleaned dataset
    out_file = Path(clean_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    return df, report


if __name__ == "__main__":
    clean_df, rep = clean_products()
    print("Cleaned products:", len(clean_df))
    print("Categories:", rep["categories"])
    print(f"Price range: INR {rep['min_price']} - INR {rep['max_price']}")
