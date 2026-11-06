"""Customer segment profiling, naming, and export for Instamart.

Assigns meaningful business archetypes and actionable marketing strategies to clusters.
Marks the segment with the highest mean monetary spend as Premium (is_premium = True).
Exports outputs/customer_segments.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def profile_segments(
    rfm_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    output_path: Path | str = "outputs/customer_segments.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile clusters, assign business names, flag premium segment, and export customer segments."""
    df = rfm_df.copy()
    df["segment_id"] = cluster_labels

    # Compute group averages to identify archetypes
    cluster_stats = df.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()

    # The cluster with the highest mean monetary value is strictly marked Premium
    premium_cluster_id = int(cluster_stats.loc[cluster_stats["mean_monetary"].idxmax(), "segment_id"])

    # Define segment mapping based on behavioral profiles:
    # 1. Highest monetary -> Premium Spenders (size ~138, monetary ~7,130, freq ~6.8)
    # 2. Lowest recency & highest frequency -> Frequent Champions (size ~419, recency ~8.8, freq ~19.4)
    # 3. Moderate recency & moderate frequency -> Occasional Regulars (size ~436, recency ~28.9, freq ~3.8)
    # 4. Highest recency & lowest frequency -> At-Risk / Hibernating (size ~182, recency ~109.0, freq ~2.2)
    segment_names = {}
    recommendations = {}

    for _, row in cluster_stats.iterrows():
        cid = int(row["segment_id"])
        if cid == premium_cluster_id:
            segment_names[cid] = "Premium High-Spenders"
            recommendations[cid] = "Exclusive high-tier perks, priority dispatch, and dedicated VIP support concierge."
        elif row["mean_frequency"] > 15:
            segment_names[cid] = "Frequent Champions"
            recommendations[cid] = "Subscription cross-sells, loyalty milestone points, and new category trial bundles."
        elif row["mean_recency"] > 80:
            segment_names[cid] = "At-Risk / Hibernating"
            recommendations[cid] = "Aggressive win-back discounts, personalized re-engagement SMS/push, and satisfaction surveys."
        else:
            segment_names[cid] = "Occasional Regulars"
            recommendations[cid] = "Category discovery coupons, mid-week delivery discounts, and weekend pantry reminders."

    cluster_stats["segment_name"] = cluster_stats["segment_id"].map(segment_names)
    cluster_stats["recommended_action"] = cluster_stats["segment_id"].map(recommendations)
    cluster_stats["is_premium"] = cluster_stats["segment_id"] == premium_cluster_id

    # Format customer segments table
    df["segment_name"] = df["segment_id"].map(segment_names)
    df["is_premium"] = df["segment_id"] == premium_cluster_id

    # Exact required columns:
    # customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium
    clean_segment_cols = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
    df_export = df[clean_segment_cols].copy()

    # Export to outputs/
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_export.to_csv(out_file, index=False)

    return df_export, cluster_stats


if __name__ == "__main__":
    from src.part_b.rfm_analysis import compute_rfm_table
    from src.part_b.clustering import evaluate_kmeans, fit_final_kmeans
    rfm_data, _ = compute_rfm_table()
    _, _, X_scaled = evaluate_kmeans(rfm_data)
    _, labels = fit_final_kmeans(X_scaled, chosen_k=4)
    cust_segs, profiles = profile_segments(rfm_data, labels)
    print("Segment Profiles:")
    print(profiles[["segment_id", "segment_name", "size", "mean_recency", "mean_frequency", "mean_monetary", "is_premium"]])
