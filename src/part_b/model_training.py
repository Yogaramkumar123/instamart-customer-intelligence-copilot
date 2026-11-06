"""Model training, hyperparameter tuning via GridSearchCV, and evaluation for lapse prediction.

Trains:
1. Baseline: Always predict negative (not lapsed).
2. Logistic Regression: Grid search over C in {0.01, 0.1, 1, 10, 100}.
3. Random Forest: Grid search over max_depth in {3, 5, 8} and min_samples_leaf in {5, 10, 20}.

Evaluates both on untouched test set for the positive (lapsed) class.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from src.part_b.preprocessing import create_preprocessor


def evaluate_baseline(y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate baseline rule: always predict not lapsed (0)."""
    y_pred_baseline = np.zeros_like(y_test)
    acc = round(float(accuracy_score(y_test, y_pred_baseline)), 4)
    f1 = round(float(f1_score(y_test, y_pred_baseline, zero_division=0)), 4)
    p = round(float(precision_score(y_test, y_pred_baseline, zero_division=0)), 4)
    r = round(float(recall_score(y_test, y_pred_baseline, zero_division=0)), 4)

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    }


def train_and_tune_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Tuple[GridSearchCV, GridSearchCV, Dict[str, Any], Any]:
    """Train and evaluate Logistic Regression and Random Forest models with 5-fold cross validation."""
    preprocessor = create_preprocessor()

    # 1. Logistic Regression Pipeline & Grid
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_grid = GridSearchCV(
        estimator=lr_pipeline,
        param_grid={"clf__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1"
    )
    lr_grid.fit(X_train, y_train)

    # 2. Random Forest Pipeline & Grid
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid={
            "clf__max_depth": [3, 5, 8],
            "clf__min_samples_leaf": [5, 10, 20]
        },
        cv=5,
        scoring="f1"
    )
    rf_grid.fit(X_train, y_train)

    # 3. Test set evaluation on untouched test data
    y_pred_lr = lr_grid.predict(X_test)
    y_pred_rf = rf_grid.predict(X_test)

    eval_results = {
        "baseline": evaluate_baseline(y_test),
        "logistic_regression": {
            "best_params": lr_grid.best_params_,
            "best_cv_f1": round(float(lr_grid.best_score_), 4),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred_lr)), 4),
            "test_precision": round(float(precision_score(y_test, y_pred_lr)), 4),
            "test_recall": round(float(recall_score(y_test, y_pred_lr)), 4),
            "test_f1": round(float(f1_score(y_test, y_pred_lr)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist()
        },
        "random_forest": {
            "best_params": rf_grid.best_params_,
            "best_cv_f1": round(float(rf_grid.best_score_), 4),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred_rf)), 4),
            "test_precision": round(float(precision_score(y_test, y_pred_rf)), 4),
            "test_recall": round(float(recall_score(y_test, y_pred_rf)), 4),
            "test_f1": round(float(f1_score(y_test, y_pred_rf)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist()
        }
    }

    # Select the model with highest CV F1 and test performance
    if eval_results["random_forest"]["best_cv_f1"] >= eval_results["logistic_regression"]["best_cv_f1"]:
        selected_model = rf_grid.best_estimator_
        selected_name = "random_forest"
    else:
        selected_model = lr_grid.best_estimator_
        selected_name = "logistic_regression"

    eval_results["selected_model_name"] = selected_name

    return lr_grid, rf_grid, eval_results, selected_model


if __name__ == "__main__":
    from src.part_b.lapse_features import build_lapse_dataset
    from src.part_b.preprocessing import prepare_train_test_data
    X_raw, y_all, _ = build_lapse_dataset()
    X_tr, X_te, y_tr, y_te, _ = prepare_train_test_data(X_raw, y_all)
    _, _, res, sel = train_and_tune_models(X_tr, y_tr, X_te, y_te)
    print("Baseline:", res["baseline"])
    print("\nLogistic Regression:", res["logistic_regression"])
    print("\nRandom Forest:", res["random_forest"])
    print("\nSelected Model:", res["selected_model_name"])
