"""Master orchestration pipeline for Part B: Customer Segmentation and Lapse Prediction.

Executes end-to-end:
1. RFM table construction (as of 2025-10-01).
2. K-Means clustering evaluation (K=2..8) and K=4 fitting.
3. Segment profiling, naming, and customer segment export (outputs/customer_segments.csv).
4. PCA 2D projection and explained variance analysis.
5. Lapse cohort and leakage-free feature engineering (cutoff 2025-09-01).
6. Stratified 80/20 train/test partitioning and leakage-free ColumnTransformer preprocessing.
7. Hyperparameter tuning for Logistic Regression & Random Forest via 5-fold GridSearchCV.
8. Class-balance oversampling experiment on training partition.
9. Feature importance and coefficient interpretation.
10. Model persistence to outputs/lapse_model.joblib and verification.
11. Premium retention list generation to outputs/retention_list.csv.
12. Comprehensive automated checkpoint validation.
"""

from pathlib import Path
import sys
import joblib

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.part_b.rfm_analysis import compute_rfm_table
from src.part_b.clustering import evaluate_kmeans, fit_final_kmeans
from src.part_b.segment_profiling import profile_segments
from src.part_b.pca_analysis import run_pca_analysis
from src.part_b.lapse_features import build_lapse_dataset
from src.part_b.preprocessing import prepare_train_test_data
from src.part_b.model_training import train_and_tune_models
from src.part_b.class_balance import run_class_balance_experiment
from src.part_b.model_interpretation import interpret_models
from src.part_b.retention_list import generate_premium_retention_list
from src.part_b.validate import validate_part_b


def run_pipeline():
    print("=" * 60)
    print("INSTAMART COPILOT — PART B PIPELINE EXECUTION")
    print("=" * 60)

    # Verify input clean files exist
    clean_orders = REPO_ROOT / "data" / "clean" / "orders.csv"
    clean_customers = REPO_ROOT / "data" / "clean" / "customers.csv"
    clean_products = REPO_ROOT / "data" / "clean" / "products.csv"

    if not clean_orders.exists() or not clean_customers.exists():
        raise FileNotFoundError("Part A clean data files missing! Run scripts/run_part_a.py first.")

    # 1. RFM Table Construction
    print("\n[Step 1] Computing customer RFM table as of 2025-10-01...")
    rfm_df, rfm_stats = compute_rfm_table(clean_orders_path=clean_orders)
    print(f"  Customers: {rfm_stats['customer_count']}")
    print(f"  Recency Median: {rfm_stats['recency_median']} days (Range: {rfm_stats['recency_min']}-{rfm_stats['recency_max']})")
    print(f"  Frequency Mean: {rfm_stats['frequency_mean']} (Max: {rfm_stats['frequency_max']})")
    print(f"  Monetary Mean: INR {rfm_stats['monetary_mean']:,.2f} (Max: INR {rfm_stats['monetary_max']:,.2f})")

    # 2. K-Means Clustering Evaluation
    print("\n[Step 2] Evaluating K-Means clustering across K=2..8...")
    comp_df, scaler, X_scaled = evaluate_kmeans(
        rfm_df=rfm_df,
        output_dir=REPO_ROOT / "outputs" / "part_b"
    )
    for _, r in comp_df.iterrows():
        print(f"  K={int(r['K'])}: Inertia={r['inertia']:.1f}, Silhouette={r['silhouette']:.3f}")

    print("\n  Fitting final K-Means with chosen K=4...")
    km4, labels = fit_final_kmeans(X_scaled, chosen_k=4)

    # 3. Segment Profiling & Export
    print("\n[Step 3] Profiling segments and exporting outputs/customer_segments.csv...")
    cust_segs, profiles = profile_segments(
        rfm_df=rfm_df,
        cluster_labels=labels,
        output_path=REPO_ROOT / "outputs" / "customer_segments.csv"
    )
    for _, p in profiles.iterrows():
        print(f"  Segment {p['segment_id']} ('{p['segment_name']}'): Size={p['size']}, Recency={p['mean_recency']:.1f}d, Freq={p['mean_frequency']:.1f}, Monetary=INR {p['mean_monetary']:,.2f}, Premium={p['is_premium']}")

    # 4. PCA Analysis
    print("\n[Step 4] Running PCA 2-component projection on standardized RFM...")
    pca_res = run_pca_analysis(
        X_scaled=X_scaled,
        output_dir=REPO_ROOT / "outputs" / "part_b"
    )
    print(f"  PC1 Variance: {pca_res['pc1_explained_variance_ratio']:.4f}")
    print(f"  PC2 Variance: {pca_res['pc2_explained_variance_ratio']:.4f}")
    print(f"  Total Explained Variance: {pca_res['total_explained_variance_ratio']:.4f}")

    # 5. Lapse Cohort & Feature Engineering
    print("\n[Step 5] Constructing leakage-free lapse cohort and features (cutoff 2025-09-01)...")
    X_features, y_target, meta = build_lapse_dataset(
        clean_orders_path=clean_orders,
        clean_customers_path=clean_customers,
        clean_products_path=clean_products
    )
    print(f"  Cohort Size: {meta['cohort_size']} customers")
    print(f"  Lapsed Customers: {meta['lapsed_count']} ({meta['lapse_rate']}%)")

    # 6. Train-Test Splitting & Preprocessing
    print("\n[Step 6] Partitioning data (80/20 train/test) and verifying leakage-free preprocessing...")
    X_train, X_test, y_train, y_test, split_metrics = prepare_train_test_data(
        df_features=X_features,
        y=y_target
    )
    print(f"  Train split: {split_metrics['train_size']} rows ({split_metrics['train_lapsed_count']} lapsed)")
    print(f"  Test split: {split_metrics['test_size']} rows ({split_metrics['test_lapsed_count']} lapsed)")
    print(f"  Full cohort freq mean: {split_metrics['full_cohort_frequency_mean']:.4f}")
    print(f"  Train scaler freq mean: {split_metrics['train_scaler_frequency_mean']:.4f}")

    # 7. Model Training & Hyperparameter Tuning
    print("\n[Step 7] Tuning Logistic Regression and Random Forest with 5-fold CV...")
    lr_grid, rf_grid, eval_results, selected_pipeline = train_and_tune_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    base = eval_results["baseline"]
    lr_eval = eval_results["logistic_regression"]
    rf_eval = eval_results["random_forest"]

    print(f"  Baseline Accuracy: {base['accuracy']:.4f}, Baseline F1: {base['f1']:.4f}")
    print(f"  Logistic Regression Best Params: {lr_eval['best_params']}, Best CV F1: {lr_eval['best_cv_f1']:.4f}")
    print(f"    Test Accuracy: {lr_eval['test_accuracy']:.4f}, Precision: {lr_eval['test_precision']:.4f}, Recall: {lr_eval['test_recall']:.4f}, F1: {lr_eval['test_f1']:.4f}")
    print(f"  Random Forest Best Params: {rf_eval['best_params']}, Best CV F1: {rf_eval['best_cv_f1']:.4f}")
    print(f"    Test Accuracy: {rf_eval['test_accuracy']:.4f}, Precision: {rf_eval['test_precision']:.4f}, Recall: {rf_eval['test_recall']:.4f}, F1: {rf_eval['test_f1']:.4f}")
    print(f"  Selected Model: {eval_results['selected_model_name']}")

    # 8. Class-Balance Experiment
    print("\n[Step 8] Running class-balance oversampling experiment on training set...")
    best_c = lr_grid.best_params_.get("clf__C", 1.0)
    bal_res = run_class_balance_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        best_c=best_c
    )
    print(f"  Unbalanced LR: Precision={bal_res['unbalanced']['precision']:.4f}, Recall={bal_res['unbalanced']['recall']:.4f}, F1={bal_res['unbalanced']['f1']:.4f}")
    print(f"  Balanced LR:   Precision={bal_res['balanced']['precision']:.4f}, Recall={bal_res['balanced']['recall']:.4f}, F1={bal_res['balanced']['f1']:.4f}")
    print(f"  Metric Shift:  Recall change = +{bal_res['metric_shift']['recall_increase']:.4f}")

    # 9. Model Interpretation
    print("\n[Step 9] Extracting feature importances and coefficients...")
    interp = interpret_models(rf_pipeline=rf_grid.best_estimator_, lr_pipeline=lr_grid.best_estimator_)
    print("  Random Forest Top 5 Features:")
    for f in interp["rf_top_5_features"]:
        print(f"    - {f['feature']}: {f['importance']:.4f}")
    print("  Logistic Regression Key Coefficients:")
    for c in interp["lr_coefficients"]:
        print(f"    - {c['feature']}: {c['coefficient']:.4f} ({c['sign']})")

    # 10. Model Persistence
    print("\n[Step 10] Persisting selected model to outputs/lapse_model.joblib...")
    model_out = REPO_ROOT / "outputs" / "lapse_model.joblib"
    joblib.dump(selected_pipeline, model_out)
    reloaded = joblib.load(model_out)
    assert (selected_pipeline.predict(X_test) == reloaded.predict(X_test)).all()
    print("  Model reloaded and predictions verified identical.")

    # 11. Retention List Generation
    print("\n[Step 11] Generating illustrative Premium retention list to outputs/retention_list.csv...")
    ret_df, ret_meta = generate_premium_retention_list(
        selected_pipeline=selected_pipeline,
        X_test=X_test,
        y_test=y_test,
        df_features=X_features,
        customer_segments_path=REPO_ROOT / "outputs" / "customer_segments.csv",
        output_path=REPO_ROOT / "outputs" / "retention_list.csv"
    )
    print(f"  Top 10 candidates selected from {ret_meta['total_test_premium_customers']} test-set Premium customers.")
    print(f"  Customers who actually lapsed: {ret_meta['actually_lapsed_count_top10']}/10.")

    # 12. Automated Checkpoint Validation
    print("\n[Step 12] Running automated Part B validation suite...")
    validate_part_b()


if __name__ == "__main__":
    run_pipeline()
