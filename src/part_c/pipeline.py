"""End-to-end triage pipeline orchestration, comparison, and budget demo."""
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import mock_llm
from src.part_c.classifier import classify_ticket
from src.part_c.prompts import ZERO_SHOT_SYSTEM_PROMPT, FEW_SHOT_SYSTEM_PROMPT
from src.part_c.routing import load_customer_segments, route_ticket
from src.part_c.token_budget import TokenBudgetManager


def run_pipeline(
    tickets_path: str = "data/raw/tickets.csv",
    segments_path: str = "outputs/customer_segments.csv",
    mode: str = "few_shot",
    token_budget: int = 100_000,
    reset_mock: bool = True,
    log_retries: bool = False,
    retry_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run support ticket triage pipeline over all tickets in original order.
    
    Args:
        tickets_path: Path to tickets.csv (must be read with keep_default_na=False).
        segments_path: Path to customer_segments.csv.
        mode: 'zero_shot' or 'few_shot'.
        token_budget: Maximum tokens budget before cutting off model calls.
        reset_mock: Whether to call mock_llm.reset() prior to the run.
        log_retries: Whether to print retry log lines.
        retry_records: Optional list to store retry event audit records.
        
    Returns:
        (triage_df, full_eval_df)
        triage_df contains: ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used
        full_eval_df contains true_category and text for downstream metrics.
    """
    if reset_mock:
        mock_llm.reset()

    tickets_df = pd.read_csv(tickets_path, keep_default_na=False)
    segments_lookup = load_customer_segments(segments_path)
    budget_mgr = TokenBudgetManager(budget=token_budget)

    system_prompt = ZERO_SHOT_SYSTEM_PROMPT if mode == "zero_shot" else FEW_SHOT_SYSTEM_PROMPT

    results = []
    eval_rows = []

    for _, row in tickets_df.iterrows():
        tid = str(row["ticket_id"])
        cid = str(row["customer_id"])
        text = str(row["text"])
        true_cat = str(row.get("true_category", ""))

        status, raw_cat, tokens = classify_ticket(
            ticket_id=tid,
            raw_text=text,
            system_prompt=system_prompt,
            budget_manager=budget_mgr,
            log_retry=log_retries,
            retry_records=retry_records,
        )

        cat_str, seg_str, priority, route = route_ticket(
            status=status,
            category=raw_cat,
            customer_id=cid,
            segments_lookup=segments_lookup,
        )

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": cat_str,
            "segment_name": seg_str,
            "priority": priority,
            "route": route,
            "tokens_used": tokens,
        })

        eval_rows.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": cat_str,
            "true_category": true_cat,
            "tokens_used": tokens,
        })

    triage_df = pd.DataFrame(results)
    eval_df = pd.DataFrame(eval_rows)
    return triage_df, eval_df


def compute_metrics(eval_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute classification accuracy, token stats, and confusion matrix over classified tickets."""
    classified = eval_df[eval_df["status"].isin(["classified_json", "classified_keyword"])].copy()
    n_classified = len(classified)

    if n_classified == 0:
        return {
            "n_classified": 0,
            "accuracy": 0.0,
            "total_tokens": int(eval_df["tokens_used"].sum()),
            "tokens_per_classified": 0.0,
            "confusion_matrix": None,
            "errors": [],
        }

    correct = (classified["category"] == classified["true_category"]).sum()
    accuracy = correct / n_classified
    total_tokens = int(eval_df["tokens_used"].sum())
    tokens_per_ticket = total_tokens / n_classified

    cats = ["billing", "delivery", "technical", "fraud"]
    cm = pd.crosstab(
        classified["true_category"],
        classified["category"],
        rownames=["True"],
        colnames=["Predicted"],
        dropna=False,
    ).reindex(index=cats, columns=cats, fill_value=0)

    # Collect errors
    errs = classified[classified["category"] != classified["true_category"]]
    error_summary = []
    for _, err in errs.iterrows():
        error_summary.append((err["ticket_id"], err["true_category"], err["category"]))

    return {
        "n_classified": n_classified,
        "correct": int(correct),
        "accuracy": float(accuracy),
        "total_tokens": total_tokens,
        "tokens_per_classified": float(tokens_per_ticket),
        "confusion_matrix": cm,
        "errors": error_summary,
    }
