"""Clean customers dataset for Instamart Customer Intelligence & Support Copilot.

Rules:
1. Strip leading and trailing spaces from city.
2. Apply title case to city.
3. Ensure exactly six standard cities remain.
4. Normalize gender: male, female, unknown.
5. Map blank gender to unknown.
6. Treat ages outside 10–100 as missing.
7. Fill missing ages using the median age of the corresponding city.
8. Use groupby and transform for city-level median imputation.
9. Convert signup_date into datetime.
10. Export to data/clean/customers.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def clean_customers(
    raw_path: Path | str = "data/raw/customers.csv",
    clean_path: Path | str = "data/clean/customers.csv"
) -> tuple[pd.DataFrame, dict]:
    """Clean customers dataframe according to Part A specifications."""
    df = pd.read_csv(raw_path)
    report = {}

    # Initial checks
    report["initial_rows"] = len(df)

    # 1 & 2. Standardise city: strip spaces and apply title case
    raw_cities = df["city"].copy()
    df["city"] = df["city"].astype(str).str.strip().str.title()
    report["cities_standardized_changes"] = int((raw_cities != df["city"]).sum())

    # 3. Ensure exactly six standard cities remain
    unique_cities = sorted(df["city"].unique())
    expected_cities = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"]
    if sorted(unique_cities) != expected_cities:
        raise ValueError(f"Expected cities {expected_cities}, got {unique_cities}")
    report["unique_cities"] = unique_cities
    report["city_counts"] = df["city"].value_counts().to_dict()

    # 4 & 5. Normalize gender to male, female, unknown
    def normalize_gender(val):
        if pd.isna(val) or str(val).strip() == "":
            return "unknown"
        val_clean = str(val).strip().lower()
        if val_clean in ["m", "male"]:
            return "male"
        elif val_clean in ["f", "female"]:
            return "female"
        return "unknown"

    raw_genders = df["gender"].copy()
    df["gender"] = df["gender"].apply(normalize_gender)
    report["genders_normalized_changes"] = int((raw_genders != df["gender"]).sum())
    report["gender_counts"] = df["gender"].value_counts().to_dict()

    # 6. Treat ages outside 10-100 as missing
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    outside_mask = (df["age"] < 10) | (df["age"] > 100)
    report["ages_outside_range_count"] = int(outside_mask.sum())
    report["initial_missing_ages"] = int(df["age"].isna().sum())
    df.loc[outside_mask, "age"] = np.nan
    report["total_missing_before_imputation"] = int(df["age"].isna().sum())

    # 7 & 8. Fill missing ages using city-level median age via groupby and transform
    city_medians = df.groupby("city")["age"].transform("median")
    imputed_count = int(df["age"].isna().sum())
    df["age"] = df["age"].fillna(city_medians)
    report["ages_imputed_count"] = imputed_count
    df["age"] = df["age"].astype(int)

    # 9. Convert signup_date to datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    # Format signup_date as YYYY-MM-DD string for clean storage
    df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")

    # Final validations
    report["final_missing_ages"] = int(df["age"].isna().sum())
    report["min_age"] = int(df["age"].min())
    report["max_age"] = int(df["age"].max())

    if report["final_missing_ages"] != 0:
        raise ValueError(f"Expected 0 missing ages, found {report['final_missing_ages']}")
    if report["min_age"] < 18 or report["max_age"] > 60:
        raise ValueError(f"Ages must be between 18 and 60, got min={report['min_age']}, max={report['max_age']}")

    # 10. Export cleaned dataset
    out_file = Path(clean_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    return df, report


if __name__ == "__main__":
    clean_df, rep = clean_customers()
    print("Cleaned customers:", len(clean_df))
    print("City counts:", rep["city_counts"])
    print("Gender counts:", rep["gender_counts"])
    print(f"Age range: {rep['min_age']} - {rep['max_age']}, Missing: {rep['final_missing_ages']}")
