"""Execute and validate all seven SQL queries for Instamart Customer Intelligence & Support Copilot.

Queries are parsed from sql/queries.sql and executed via pd.read_sql_query against data/instamart.db.
Results are validated against reference numbers and saved to outputs/part_a/.
"""

from pathlib import Path
import re
import sqlite3
import pandas as pd


def load_queries(sql_file_path: Path | str = "sql/queries.sql") -> dict[str, str]:
    """Parse individual SQL queries from sql/queries.sql based on [Q1]..[Q7] tags."""
    content = Path(sql_file_path).read_text(encoding="utf-8")
    pattern = r"--\s*\[(Q[1-7])\][^\n]*\n(.*?)(?=(?:--\s*\[Q[1-7]\])|\Z)"
    matches = re.findall(pattern, content, flags=re.DOTALL)
    queries = {}
    for q_id, q_body in matches:
        # Strip trailing comments or empty lines
        cleaned_body = "\n".join(
            line for line in q_body.strip().splitlines()
            if not line.strip().startswith("--")
        ).strip()
        queries[q_id] = cleaned_body
    if len(queries) != 7:
        raise ValueError(f"Expected 7 queries, parsed {len(queries)} from {sql_file_path}")
    return queries


def run_sql_analysis(
    db_path: Path | str = "data/instamart.db",
    sql_file_path: Path | str = "sql/queries.sql",
    output_dir: Path | str = "outputs/part_a"
) -> dict:
    """Execute all seven SQL queries and return validation and analysis results."""
    queries = load_queries(sql_file_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    results = {}
    try:
        # Execute Q1
        df_q1 = pd.read_sql_query(queries["Q1"], conn)
        df_q1.to_csv(out_dir / "q1_result.csv", index=False)
        results["Q1"] = {
            "row_count": int(len(df_q1)),
            "customer_ids": list(df_q1["customer_id"].values),
            "expected_count": 30
        }

        # Execute Q2
        df_q2 = pd.read_sql_query(queries["Q2"], conn)
        df_q2.to_csv(out_dir / "q2_result.csv", index=False)
        blr_elec = df_q2[(df_q2["city"] == "Bengaluru") & (df_q2["category"] == "Electronics")]
        blr_elec_rev = float(blr_elec["revenue"].iloc[0]) if len(blr_elec) > 0 else 0.0
        results["Q2"] = {
            "row_count": int(len(df_q2)),
            "grand_total": round(float(df_q2["revenue"].sum()), 2),
            "bengaluru_electronics": round(blr_elec_rev, 2)
        }

        # Execute Q3
        df_q3 = pd.read_sql_query(queries["Q3"], conn)
        df_q3.to_csv(out_dir / "q3_result.csv", index=False)
        results["Q3"] = {
            "row_count": int(len(df_q3)),
            "product_ids": list(df_q3["product_id"].values),
            "expected_count": 11
        }

        # Execute Q4
        df_q4 = pd.read_sql_query(queries["Q4"], conn)
        df_q4.to_csv(out_dir / "q4_result.csv", index=False)
        results["Q4"] = {
            "row_count": int(len(df_q4)),
            "customer_ids": list(df_q4["customer_id"].values),
            "expected_count": 25
        }

        # Execute Q5
        df_q5 = pd.read_sql_query(queries["Q5"], conn)
        df_q5.to_csv(out_dir / "q5_result.csv", index=False)
        results["Q5"] = {
            "row_count": int(len(df_q5)),
            "average_spend": float(df_q5["avg_spend"].iloc[0]),
            "expected_count": 550
        }

        # Execute Q6
        df_q6 = pd.read_sql_query(queries["Q6"], conn)
        df_q6.to_csv(out_dir / "q6_result.csv", index=False)
        top_by_city = df_q6[df_q6["spend_rank"] == 1].set_index("city")["customer_id"].to_dict()
        top_spend_by_city = df_q6[df_q6["spend_rank"] == 1].set_index("city")["total_spend"].to_dict()
        diff_rank_count = int((df_q6["spend_rank"] != df_q6["spend_row_number"]).sum())
        results["Q6"] = {
            "top_by_city": top_by_city,
            "top_spend_by_city": top_spend_by_city,
            "rank_row_number_diff_count": diff_rank_count
        }

        # Execute Q7
        df_q7 = pd.read_sql_query(queries["Q7"], conn)
        df_q7.to_csv(out_dir / "q7_result.csv", index=False)
        results["Q7"] = {
            "row_count": int(len(df_q7)),
            "final_cumulative_revenue": round(float(df_q7["cumulative_revenue"].iloc[-1]), 2),
            "expected_count": 183
        }

    finally:
        conn.close()

    return results


if __name__ == "__main__":
    res = run_sql_analysis()
    print("Q1 count:", res["Q1"]["row_count"])
    print("Q2 count:", res["Q2"]["row_count"], "Grand total:", res["Q2"]["grand_total"])
    print("Q3 count:", res["Q3"]["row_count"])
    print("Q4 count:", res["Q4"]["row_count"])
    print("Q5 count:", res["Q5"]["row_count"], "Avg spend:", res["Q5"]["average_spend"])
    print("Q6 diff count:", res["Q6"]["rank_row_number_diff_count"], "Tops:", res["Q6"]["top_by_city"])
    print("Q7 count:", res["Q7"]["row_count"], "Final cum rev:", res["Q7"]["final_cumulative_revenue"])
