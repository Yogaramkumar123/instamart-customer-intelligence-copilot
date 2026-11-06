"""Model interpretation and feature importance extraction for lapse prediction.

Extracts:
1. Top 5 feature importances from best Random Forest.
2. Meaningful feature coefficients and signs from best Logistic Regression.
Provides business interpretations for retention decision-making.
"""

from typing import Dict, Any, List
import pandas as pd
from sklearn.pipeline import Pipeline


def interpret_models(
    rf_pipeline: Pipeline,
    lr_pipeline: Pipeline
) -> Dict[str, Any]:
    """Extract and interpret feature importances and coefficients from tuned models."""
    prep = rf_pipeline.named_steps["preprocessor"]
    feature_names = prep.get_feature_names_out()

    # Clean prefixes from feature names (e.g., 'num__frequency' -> 'frequency')
    clean_names = [f.split("__")[-1] for f in feature_names]

    # 1. Random Forest Feature Importances
    rf_clf = rf_pipeline.named_steps["clf"]
    rf_importances = list(zip(clean_names, rf_clf.feature_importances_))
    rf_importances.sort(key=lambda x: x[1], reverse=True)

    rf_top5 = []
    rf_interpretations = {
        "monetary": "Cumulative spend captures customer economic commitment; higher spend reflects broader basket adoption.",
        "frequency": "Order volume is the primary behavioral driver of retention; regular shoppers exhibit deeply entrenched habits.",
        "recency": "Days elapsed since the last order indicates dormancy; customers with longer gaps are approaching lapse thresholds.",
        "average_delivery_minutes": "Delivery latency directly impacts customer satisfaction; slower service erodes platform loyalty.",
        "average_discount_pct": "Discount sensitivity highlights deal-seekers who may churn when promotional subsidies subside."
    }

    for name, imp in rf_importances[:5]:
        rf_top5.append({
            "feature": name,
            "importance": round(float(imp), 4),
            "business_reading": rf_interpretations.get(
                name,
                f"Feature {name} contributes {round(float(imp)*100, 2)}% toward tree split purity."
            )
        })

    # 2. Logistic Regression Coefficients
    lr_clf = lr_pipeline.named_steps["clf"]
    lr_coefs = list(zip(clean_names, lr_clf.coef_[0]))
    lr_coefs.sort(key=lambda x: abs(x[1]), reverse=True)

    lr_meaningful = []
    lr_interpretations = {
        "frequency": "Strong negative coefficient: each standard deviation increase in order frequency dramatically reduces lapse odds.",
        "favourite_category_Electronics": "Negative coefficient: electronics buyers demonstrate high affinity and higher retention.",
        "monetary": "Positive coefficient when controlling for frequency: infrequent single high-value purchases carry higher lapse risk than steady lower-value orders.",
        "recency": "Positive coefficient: longer gaps between orders directly increase the log-odds of a customer lapsing."
    }

    for name, coef in lr_coefs[:4]:
        sign = "positive" if coef > 0 else "negative"
        lr_meaningful.append({
            "feature": name,
            "coefficient": round(float(coef), 4),
            "sign": sign,
            "business_reading": lr_interpretations.get(
                name,
                f"A {sign} relationship indicating that higher values {'increase' if coef > 0 else 'decrease'} lapse probability."
            )
        })

    return {
        "rf_top_5_features": rf_top5,
        "lr_coefficients": lr_meaningful
    }


if __name__ == "__main__":
    from src.part_b.lapse_features import build_lapse_dataset
    from src.part_b.preprocessing import prepare_train_test_data
    from src.part_b.model_training import train_and_tune_models
    X_raw, y_all, _ = build_lapse_dataset()
    X_tr, X_te, y_tr, y_te, _ = prepare_train_test_data(X_raw, y_all)
    lr_g, rf_g, _, _ = train_and_tune_models(X_tr, y_tr, X_te, y_te)
    interp = interpret_models(rf_g.best_estimator_, lr_g.best_estimator_)
    print("Random Forest Top 5 Features:")
    for f in interp["rf_top_5_features"]:
        print(f"  {f['feature']}: {f['importance']} -> {f['business_reading']}")
    print("\nLogistic Regression Coefficients:")
    for c in interp["lr_coefficients"]:
        print(f"  {c['feature']}: {c['coefficient']} ({c['sign']}) -> {c['business_reading']}")
