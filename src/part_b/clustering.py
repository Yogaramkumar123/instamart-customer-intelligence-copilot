"""K-Means clustering on standardized RFM features.

Evaluates K=2..8 with KMeans(n_init=10, random_state=42).
Calculates inertia and silhouette score for each K.
Selects optimal K from {3, 4, 5} based on elbow behavior, silhouette score,
and business interpretability.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_kmeans(
    rfm_df: pd.DataFrame,
    output_dir: Path | str = "outputs/part_b"
) -> tuple[pd.DataFrame, StandardScaler, np.ndarray]:
    """Run K-Means for K=2..8 on standardized RFM features and report metrics."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rfm_features = rfm_df[["recency", "frequency", "monetary"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_features)

    records = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia = round(float(km.inertia_), 1)
        sil = round(float(silhouette_score(X_scaled, labels)), 3)
        records.append({
            "K": k,
            "inertia": inertia,
            "silhouette": sil
        })

    comparison_df = pd.DataFrame(records)
    comparison_df.to_csv(out_dir / "kmeans_comparison.csv", index=False)
    return comparison_df, scaler, X_scaled


def fit_final_kmeans(
    X_scaled: np.ndarray,
    chosen_k: int = 4
) -> tuple[KMeans, np.ndarray]:
    """Fit final K-Means model with chosen K on standardized features."""
    km = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    return km, labels


if __name__ == "__main__":
    from src.part_b.rfm_analysis import compute_rfm_table
    rfm_data, _ = compute_rfm_table()
    comp_df, sc, X_sc = evaluate_kmeans(rfm_data)
    print("K-Means Comparison (K=2..8):")
    print(comp_df)
    final_km, cluster_labels = fit_final_kmeans(X_sc, chosen_k=4)
    print(f"\nFinal K=4 Inertia: {final_km.inertia_:.1f}")
