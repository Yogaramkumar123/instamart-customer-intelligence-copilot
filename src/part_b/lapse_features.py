"""Lapse cohort definition and leakage-free feature engineering for Instamart.

Temporal Cutoff: 2025-09-01 00:00:00.
Cohort: Customers with at least one completed order strictly before cutoff.
Label: lapsed = 1 if no completed order in September 2025, else 0.

Strict Leakage Guards:
- Only pre-cutoff orders (< 2025-09-01) are used for feature construction.
- Customer segment labels/IDs from whole-window RFM are NEVER included in features.
- Zero missing values confirmed programmatically.
"""

from pathlib import Path
import pandas as pd
import numpy as np


LAPSE_CUTOFF = pd.Timestamp("2025-09-01 00:00:00")


def build_lapse_dataset(
    clean_orders_path: Path | str = "data/clean/orders.csv",
    clean_customers_path: Path | str = "data/clean/customers.csv",
    clean_products_path: Path | str = "data/clean/products.csv",
    cutoff: pd.Timestamp = LAPSE_CUTOFF
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Construct leakage-free feature matrix X and target y for lapse prediction."""
    df_orders = pd.read_csv(clean_orders_path)
    df_orders["order_ts"] = pd.to_datetime(df_orders["order_ts"])
    df_customers = pd.read_csv(clean_customers_path)
    df_products = pd.read_csv(clean_products_path)

    # 1. Temporal split
    pre_cutoff_orders = df_orders[df_orders["order_ts"] < cutoff].copy()
    post_cutoff_orders = df_orders[df_orders["order_ts"] >= cutoff].copy()

    # Completed orders before cutoff
    pre_completed = pre_cutoff_orders[pre_cutoff_orders["status"] == "completed"].copy()
    pre_completed = pre_completed.merge(
        df_products[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    # Cohort: customers with >= 1 completed order before cutoff
    cohort_customers = sorted(pre_completed["customer_id"].unique())
    cohort_size = len(cohort_customers)

    # Target label: active in September 2025?
    sep_completed = post_cutoff_orders[post_cutoff_orders["status"] == "completed"]
    sep_active_cids = set(sep_completed["customer_id"].unique())

    # lapsed = 1 if NOT active in September, else 0
    y = np.array([1 if cid not in sep_active_cids else 0 for cid in cohort_customers])
    lapsed_count = int(np.sum(y))
    lapse_rate = round(float(lapsed_count / cohort_size * 100.0), 2)

    # 2. Leakage-free feature construction
    customers_lookup = df_customers.set_index("customer_id")
    feature_rows = []

    for cid in cohort_customers:
        c_comp = pre_completed[pre_completed["customer_id"] == cid]
        c_all = pre_cutoff_orders[pre_cutoff_orders["customer_id"] == cid]

        # Recency: days from cutoff to last completed order before cutoff
        last_order_ts = c_comp["order_ts"].max()
        recency = (cutoff - last_order_ts).days

        # Frequency: distinct completed orders before cutoff
        frequency = int(c_comp["order_id"].nunique())

        # Monetary: sum of completed order amounts before cutoff
        monetary = float(c_comp["order_amount"].sum())

        # Average discount: mean discount_pct over completed orders before cutoff
        avg_discount = float(c_comp["discount_pct"].mean())

        # Cancellation rate: cancelled / total final orders before cutoff
        n_cancelled = int((c_all["status"] == "cancelled").sum())
        n_total = len(c_all)
        cancellation_rate = float(n_cancelled / n_total) if n_total > 0 else 0.0

        # Average delivery: mean delivery_minutes among completed orders before cutoff
        avg_delivery = float(c_comp["delivery_minutes"].mean())

        # Favourite category: mode product category from completed orders before cutoff
        mode_cat_series = c_comp["category"].mode()
        favourite_category = str(mode_cat_series.iloc[0]) if len(mode_cat_series) > 0 else "Unknown"

        # Demographics and tenure from clean customers table
        c_demog = customers_lookup.loc[cid]
        age = int(c_demog["age"])
        signup_dt = pd.to_datetime(c_demog["signup_date"])
        tenure_days = int((cutoff - signup_dt).days)
        city = str(c_demog["city"])
        gender = str(c_demog["gender"])

        feature_rows.append({
            "customer_id": cid,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "average_discount_pct": avg_discount,
            "cancellation_rate": cancellation_rate,
            "average_delivery_minutes": avg_delivery,
            "age": age,
            "tenure_days": tenure_days,
            "favourite_category": favourite_category,
            "city": city,
            "gender": gender
        })

    df_features = pd.DataFrame(feature_rows)

    # Assertions ensuring zero leakage and integrity
    assert "segment_id" not in df_features.columns, "Segment ID must not leak into features"
    assert "segment_name" not in df_features.columns, "Segment name must not leak into features"
    assert "is_premium" not in df_features.columns, "Premium flag must not leak into features"
    assert df_features.isna().sum().sum() == 0, "Feature matrix contains unexpected missing values"

    metadata = {
        "cohort_size": cohort_size,
        "lapsed_count": lapsed_count,
        "lapse_rate": lapse_rate,
        "cutoff_date": str(cutoff.date())
    }

    return df_features, y, metadata


if __name__ == "__main__":
    X_df, y_target, meta = build_lapse_dataset()
    print("Lapse Dataset Metadata:", meta)
    print("Features shape:", X_df.shape)
    print("Features sample:")
    print(X_df.head(2))
