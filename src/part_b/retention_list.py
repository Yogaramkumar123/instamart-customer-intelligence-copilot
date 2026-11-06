"""Generates the illustrative Premium retention candidate list for Part B.

Ranks test-set customers belonging to the Premium segment by their predicted lapse probability.
Exports outputs/retention_list.csv with columns:
customer_id, lapse_probability, actually_lapsed.

IMPORTANT LEAKAGE DISCLOSURE:
The Premium flag is derived from whole-window RFM (including September), which overlaps
with the lapse evaluation window. Therefore, this retention list is an illustrative operational
demonstration rather than a leakage-free predictive evaluation.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def generate_premium_retention_list(
    selected_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    df_features: pd.DataFrame,
    customer_segments_path: Path | str = "outputs/customer_segments.csv",
    output_path: Path | str = "outputs/retention_list.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Rank test-set Premium customers by predicted lapse probability and export top 10."""
    # Predict probabilities for class 1 (lapsed)
    probs = selected_pipeline.predict_proba(X_test)[:, 1]

    # Map test row index back to customer_id from df_features
    test_indices = X_test.index
    test_cids = df_features.loc[test_indices, "customer_id"].values

    test_results_df = pd.DataFrame({
        "customer_id": test_cids,
        "lapse_probability": [round(float(p), 4) for p in probs],
        "actually_lapsed": y_test.tolist()
    })

    # Merge with customer segments to find Premium customers
    cust_segs = pd.read_csv(customer_segments_path)
    merged = test_results_df.merge(
        cust_segs[["customer_id", "is_premium", "segment_name"]],
        on="customer_id",
        how="inner"
    )

    # Filter for Premium segment only
    premium_test_customers = merged[merged["is_premium"] == True].copy()

    # Rank by lapse probability descending
    ranked_premium = premium_test_customers.sort_values(
        by="lapse_probability",
        ascending=False
    ).reset_index(drop=True)

    # Select top 10 (or all if < 10)
    top_retention = ranked_premium.head(10)[["customer_id", "lapse_probability", "actually_lapsed"]].copy()

    # Export to CSV
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    top_retention.to_csv(out_file, index=False)

    summary = {
        "total_test_premium_customers": int(len(premium_test_customers)),
        "retention_list_size": int(len(top_retention)),
        "actually_lapsed_count_top10": int(top_retention["actually_lapsed"].sum()),
        "top_lapse_probability": float(top_retention["lapse_probability"].iloc[0]) if len(top_retention) > 0 else 0.0,
        "leakage_warning": (
            "The Premium flag is generated from whole-window RFM analysis, so this list is "
            "an illustrative operational prioritization and not a leakage-free evaluation."
        )
    }

    return top_retention, summary


if __name__ == "__main__":
    from src.part_b.lapse_features import build_lapse_dataset
    from src.part_b.preprocessing import prepare_train_test_data
    from src.part_b.model_training import train_and_tune_models
    X_raw, y_all, _ = build_lapse_dataset()
    X_tr, X_te, y_tr, y_te, _ = prepare_train_test_data(X_raw, y_all)
    _, _, _, best_pipe = train_and_tune_models(X_tr, y_tr, X_te, y_te)
    ret_df, meta = generate_premium_retention_list(best_pipe, X_te, y_te, X_raw)
    print("Retention List:")
    print(ret_df)
    print("\nMetadata:", meta)
