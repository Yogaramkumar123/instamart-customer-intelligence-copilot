"""Class-balance experiment via training-only random oversampling.

Procedure:
1. Retains all original training rows.
2. Oversamples minority (lapsed) training rows with replacement (random_state=42)
   until both classes equal 570.
3. Test set remains untouched.
4. Refits best Logistic Regression model and compares precision, recall, and F1.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.part_b.preprocessing import create_preprocessor


def run_class_balance_experiment(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    best_c: float = 1.0
) -> Dict[str, Any]:
    """Execute training-only oversampling experiment and evaluate impact on test metrics."""
    # 1. Unbalanced baseline Logistic Regression
    unbal_pipe = Pipeline([
        ("prep", create_preprocessor()),
        ("clf", LogisticRegression(C=best_c, max_iter=1000, random_state=42))
    ])
    unbal_pipe.fit(X_train, y_train)
    y_pred_unbal = unbal_pipe.predict(X_test)

    p_unbal = round(float(precision_score(y_test, y_pred_unbal)), 4)
    r_unbal = round(float(recall_score(y_test, y_pred_unbal)), 4)
    f_unbal = round(float(f1_score(y_test, y_pred_unbal)), 4)
    cm_unbal = confusion_matrix(y_test, y_pred_unbal).tolist()

    # 2. Oversample training set strictly
    rng = np.random.RandomState(42)
    lapsed_idx = np.where(y_train == 1)[0]
    non_lapsed_count = int(np.sum(y_train == 0))
    n_needed = non_lapsed_count - len(lapsed_idx)

    sampled_duplicates = rng.choice(lapsed_idx, size=n_needed, replace=True)
    all_train_indices = np.concatenate([np.arange(len(y_train)), sampled_duplicates])

    X_train_bal = X_train.iloc[all_train_indices].reset_index(drop=True)
    y_train_bal = y_train[all_train_indices]

    train_class_counts = [int(np.sum(y_train_bal == 0)), int(np.sum(y_train_bal == 1))]

    # 3. Refit Logistic Regression on balanced training set
    bal_pipe = Pipeline([
        ("prep", create_preprocessor()),
        ("clf", LogisticRegression(C=best_c, max_iter=1000, random_state=42))
    ])
    bal_pipe.fit(X_train_bal, y_train_bal)
    y_pred_bal = bal_pipe.predict(X_test)

    p_bal = round(float(precision_score(y_test, y_pred_bal)), 4)
    r_bal = round(float(recall_score(y_test, y_pred_bal)), 4)
    f_bal = round(float(f1_score(y_test, y_pred_bal)), 4)
    cm_bal = confusion_matrix(y_test, y_pred_bal).tolist()

    return {
        "unbalanced": {
            "precision": p_unbal,
            "recall": r_unbal,
            "f1": f_unbal,
            "confusion_matrix": cm_unbal
        },
        "balanced": {
            "precision": p_bal,
            "recall": r_bal,
            "f1": f_bal,
            "confusion_matrix": cm_bal,
            "balanced_train_counts": train_class_counts
        },
        "metric_shift": {
            "recall_increase": round(float(r_bal - r_unbal), 4),
            "precision_change": round(float(p_bal - p_unbal), 4),
            "f1_change": round(float(f_bal - f_unbal), 4)
        }
    }


if __name__ == "__main__":
    from src.part_b.lapse_features import build_lapse_dataset
    from src.part_b.preprocessing import prepare_train_test_data
    X_raw, y_all, _ = build_lapse_dataset()
    X_tr, X_te, y_tr, y_te, _ = prepare_train_test_data(X_raw, y_all)
    exp_res = run_class_balance_experiment(X_tr, y_tr, X_te, y_te)
    print("Class Balance Experiment Results:")
    print("  Unbalanced:", exp_res["unbalanced"])
    print("  Balanced:", exp_res["balanced"])
    print("  Shift:", exp_res["metric_shift"])
