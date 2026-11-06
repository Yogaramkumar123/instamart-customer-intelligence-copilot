"""Comprehensive validation suite for Part B: Customer Segmentation and Lapse Prediction.

Asserts:
1. RFM table benchmarks (customer count, recency, frequency, monetary).
2. K-Means clustering comparisons and silhouette scores.
3. K=4 segment profiling and single Premium flag assignment.
4. PCA explained variance ratios (PC1, PC2, Total).
5. Lapse cohort and split sizes (1,156 cohort, 443 lapsed, 924 train, 232 test).
6. Zero feature leakage from post-cutoff window or segment labels.
7. Model tuning metrics, baseline, and test F1 >= 0.55.
8. Class-balance experiment verification (570 balanced train count, recall increase).
9. Joblib persistence and exact prediction reproducibility.
10. Premium retention list schema, ranking, and size.

Prints 'ALL PART B CHECKPOINTS PASSED' upon complete success.
"""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

from src.part_b.rfm_analysis import compute_rfm_table
from src.part_b.clustering import evaluate_kmeans, fit_final_kmeans
from src.part_b.segment_profiling import profile_segments
from src.part_b.pca_analysis import run_pca_analysis
from src.part_b.lapse_features import build_lapse_dataset
from src.part_b.preprocessing import prepare_train_test_data
from src.part_b.model_training import train_and_tune_models
from src.part_b.class_balance import run_class_balance_experiment
from src.part_b.retention_list import generate_premium_retention_list


def assert_checkpoint(name: str, actual, expected, tol: float | None = None):
    """Assert actual matches expected within tolerance, raising informative errors on failure."""
    if tol is not None:
        if abs(actual - expected) > tol:
            print(f"FAILED CHECKPOINT: [{name}] Expected {expected} (+/- {tol}), got {actual}")
            raise AssertionError(f"Checkpoint [{name}] failed: Expected {expected}, got {actual}")
    else:
        if actual != expected:
            print(f"FAILED CHECKPOINT: [{name}] Expected {expected}, got {actual}")
            raise AssertionError(f"Checkpoint [{name}] failed: Expected {expected}, got {actual}")
    print(f"  [PASS] {name}: {actual}")


def validate_part_b() -> bool:
    """Execute complete validation suite for Part B."""
    print("==================================================")
    print("STARTING PART B COMPREHENSIVE VALIDATION")
    print("==================================================")

    # 1. RFM Table Verification
    print("\n--- 1. Verifying RFM Table Checkpoints ---")
    rfm_df, rfm_stats = compute_rfm_table()
    assert_checkpoint("RFM Customer Count", rfm_stats["customer_count"], 1175)
    assert_checkpoint("Recency Median", rfm_stats["recency_median"], 19.0)
    assert_checkpoint("Recency Minimum", rfm_stats["recency_min"], 0)
    assert_checkpoint("Recency Maximum", rfm_stats["recency_max"], 180)
    assert_checkpoint("Frequency Mean", rfm_stats["frequency_mean"], 9.46, tol=0.02)
    assert_checkpoint("Frequency Maximum", rfm_stats["frequency_max"], 33)
    assert_checkpoint("Monetary Mean (INR)", rfm_stats["monetary_mean"], 2552.06, tol=0.05)
    assert_checkpoint("Monetary Maximum (INR)", rfm_stats["monetary_max"], 14766.55, tol=0.05)

    # 2. K-Means Comparison Table Verification
    print("\n--- 2. Verifying K-Means Clustering Checkpoints ---")
    comp_df, scaler, X_scaled = evaluate_kmeans(rfm_df)
    reference_kmeans = {
        2: (1820.8, 0.465),
        3: (1268.2, 0.445),
        4: (743.1, 0.526),
        5: (617.4, 0.481),
        6: (503.2, 0.482),
        7: (408.5, 0.416),
        8: (366.5, 0.412)
    }
    for _, row in comp_df.iterrows():
        k = int(row["K"])
        ref_inertia, ref_sil = reference_kmeans[k]
        # Inertia tolerance: +/- 5%
        assert_checkpoint(f"K={k} Inertia", row["inertia"], ref_inertia, tol=ref_inertia * 0.05)
        # Silhouette tolerance: +/- 0.05
        assert_checkpoint(f"K={k} Silhouette", row["silhouette"], ref_sil, tol=0.05)

    # 3. K=4 Segment Profiling and Customer Segments Output
    print("\n--- 3. Verifying Segment Profiling & Customer Segments ---")
    km4, labels = fit_final_kmeans(X_scaled, chosen_k=4)
    cust_segs, profiles = profile_segments(rfm_df, labels)

    assert_checkpoint("Customer Segments Row Count", len(cust_segs), 1175)
    assert_checkpoint("Unique Segment Count", cust_segs["segment_id"].nunique(), 4)
    assert_checkpoint("Exactly One Premium Cluster", int(profiles["is_premium"].sum()), 1)

    prem_profile = profiles[profiles["is_premium"] == True].iloc[0]
    max_monetary_profile = profiles.loc[profiles["mean_monetary"].idxmax()]
    assert_checkpoint(
        "Premium Cluster is Highest Mean Monetary",
        prem_profile["segment_id"],
        max_monetary_profile["segment_id"]
    )
    assert_checkpoint("Premium Segment Size (~138)", int(prem_profile["size"]), 138, tol=5)
    assert_checkpoint("Premium Segment Mean Monetary (~7130)", round(float(prem_profile["mean_monetary"]), 2), 7130.0, tol=150.0)

    # 4. PCA Analysis Verification
    print("\n--- 4. Verifying PCA Explained Variance ---")
    pca_res = run_pca_analysis(X_scaled)
    assert_checkpoint("PCA PC1 Variance Ratio", pca_res["pc1_explained_variance_ratio"], 0.634, tol=0.02)
    assert_checkpoint("PCA PC2 Variance Ratio", pca_res["pc2_explained_variance_ratio"], 0.218, tol=0.02)
    assert_checkpoint("PCA Total Variance Ratio", pca_res["total_explained_variance_ratio"], 0.852, tol=0.02)

    # 5. Lapse Cohort & Leakage-Free Dataset Verification
    print("\n--- 5. Verifying Lapse Cohort and Leakage Guards ---")
    X_features, y_target, meta = build_lapse_dataset()
    assert_checkpoint("Cohort Size", meta["cohort_size"], 1156)
    assert_checkpoint("Lapsed Count", meta["lapsed_count"], 443)
    assert_checkpoint("Lapse Rate (%)", meta["lapse_rate"], 38.32, tol=0.05)
    assert_checkpoint("No Missing Values in Features", int(X_features.isna().sum().sum()), 0)
    assert_checkpoint("Segment ID Not in Features", "segment_id" in X_features.columns, False)
    assert_checkpoint("Segment Name Not in Features", "segment_name" in X_features.columns, False)
    assert_checkpoint("Premium Flag Not in Features", "is_premium" in X_features.columns, False)

    # 6. Train/Test Preprocessing Verification
    print("\n--- 6. Verifying Preprocessing and Train/Test Partitions ---")
    X_train, X_test, y_train, y_test, split_metrics = prepare_train_test_data(X_features, y_target)
    assert_checkpoint("Train Rows Count", split_metrics["train_size"], 924)
    assert_checkpoint("Test Rows Count", split_metrics["test_size"], 232)
    assert_checkpoint("Train Lapsed Count", split_metrics["train_lapsed_count"], 354)
    assert_checkpoint("Test Lapsed Count", split_metrics["test_lapsed_count"], 89)

    # 7. Model Tuning & Evaluation Verification
    print("\n--- 7. Verifying Model Tuning and Baseline ---")
    lr_grid, rf_grid, eval_results, selected_pipeline = train_and_tune_models(
        X_train, y_train, X_test, y_test
    )
    base_eval = eval_results["baseline"]
    assert_checkpoint("Baseline Accuracy", base_eval["accuracy"], 0.616, tol=0.01)
    assert_checkpoint("Baseline Lapsed F1", base_eval["f1"], 0.0)

    lr_res = eval_results["logistic_regression"]
    rf_res = eval_results["random_forest"]

    assert_checkpoint("LR Best Parameters Present", "clf__C" in lr_res["best_params"], True)
    assert_checkpoint("LR Test F1 >= 0.55", bool(lr_res["test_f1"] >= 0.55), True)
    assert_checkpoint("RF Best Parameters Present", "clf__max_depth" in rf_res["best_params"], True)
    assert_checkpoint("RF Test F1 >= 0.55", bool(rf_res["test_f1"] >= 0.55), True)

    # 8. Class Balance Experiment Verification
    print("\n--- 8. Verifying Class-Balance Experiment ---")
    best_c_val = lr_grid.best_params_.get("clf__C", 1.0)
    balance_res = run_class_balance_experiment(X_train, y_train, X_test, y_test, best_c=best_c_val)
    balanced_counts = balance_res["balanced"]["balanced_train_counts"]
    assert_checkpoint("Balanced Training Class Counts", balanced_counts, [570, 570])
    assert_checkpoint("Balanced Recall Jump (> 0.15 increase)", bool(balance_res["metric_shift"]["recall_increase"] >= 0.15), True)

    # 9. Model Persistence and Reproducibility Verification
    print("\n--- 9. Verifying Model Persistence (joblib) ---")
    model_path = Path("outputs/lapse_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, model_path)
    assert_checkpoint("Model File Exists", model_path.exists(), True)

    reloaded_pipeline = joblib.load(model_path)
    orig_preds = selected_pipeline.predict(X_test)
    reloaded_preds = reloaded_pipeline.predict(X_test)
    assert_checkpoint("Reloaded Model Predictions Match Original", bool(np.array_equal(orig_preds, reloaded_preds)), True)

    # 10. Premium Retention List Verification
    print("\n--- 10. Verifying Premium Retention List ---")
    retention_df, ret_meta = generate_premium_retention_list(
        selected_pipeline=selected_pipeline,
        X_test=X_test,
        y_test=y_test,
        df_features=X_features
    )
    ret_file = Path("outputs/retention_list.csv")
    assert_checkpoint("Retention List File Exists", ret_file.exists(), True)
    assert_checkpoint("Retention List Row Count (<= 10)", bool(len(retention_df) <= 10), True)
    expected_ret_cols = ["customer_id", "lapse_probability", "actually_lapsed"]
    assert_checkpoint("Retention List Columns", list(retention_df.columns), expected_ret_cols)
    assert_checkpoint("Retention List Probabilities Sorted Descending", bool(retention_df["lapse_probability"].is_monotonic_decreasing), True)

    print("\n==================================================")
    print("ALL PART B CHECKPOINTS PASSED")
    print("==================================================")
    return True


if __name__ == "__main__":
    try:
        success = validate_part_b()
        if not success:
            sys.exit(1)
    except Exception as exc:
        print(f"\n[PART B VALIDATION FAILED]: {exc}", file=sys.stderr)
        sys.exit(1)
