"""Principal Component Analysis (PCA) on standardized RFM feature matrix.

Projects 3D RFM space onto 2 principal components and reports explained variance ratios.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def run_pca_analysis(
    X_scaled: np.ndarray,
    output_dir: Path | str = "outputs/part_b"
) -> dict:
    """Run PCA with 2 components on scaled RFM matrix and report variance ratios."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    evr = pca.explained_variance_ratio_
    pc1_var = round(float(evr[0]), 4)
    pc2_var = round(float(evr[1]), 4)
    total_var = round(float(evr.sum()), 4)

    results = {
        "pc1_explained_variance_ratio": pc1_var,
        "pc2_explained_variance_ratio": pc2_var,
        "total_explained_variance_ratio": total_var,
        "components": pca.components_.tolist()
    }

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "pca_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    from src.part_b.rfm_analysis import compute_rfm_table
    from src.part_b.clustering import evaluate_kmeans
    rfm_data, _ = compute_rfm_table()
    _, _, X_sc = evaluate_kmeans(rfm_data)
    res = run_pca_analysis(X_sc)
    print("PCA Results:")
    print(f"  PC1 Explained Variance: {res['pc1_explained_variance_ratio']}")
    print(f"  PC2 Explained Variance: {res['pc2_explained_variance_ratio']}")
    print(f"  Total Explained Variance: {res['total_explained_variance_ratio']}")
