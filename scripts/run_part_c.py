"""Run full Part C: Support Ticket Triage with Guardrails pipeline."""
import os
import sys

# Ensure repository root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.part_c.pipeline import run_pipeline, compute_metrics
from src.part_c.pii_redaction import audit_corpus_pii, verify_no_pii_remaining
from src.part_c.validate import validate_part_c
import pandas as pd


def main() -> None:
    print("=" * 70)
    print("STARTING PART C: SUPPORT TICKET TRIAGE WITH GUARDRAILS")
    print("=" * 70)

    os.makedirs("outputs", exist_ok=True)
    tickets_path = "data/raw/tickets.csv"
    segments_path = "outputs/customer_segments.csv"
    budget_demo_path = "outputs/triage_budget_demo.csv"
    triage_results_path = "outputs/triage_results.csv"

    # 1. PII Audit
    print("\n--- 1. AUDITING INPUT GUARDRAILS & PII REDACTION ---")
    tickets_df = pd.read_csv(tickets_path, keep_default_na=False)
    redacted_texts, pii_counts = audit_corpus_pii(tickets_df["text"].tolist())
    print(f"Total tickets ingested: {len(tickets_df)}")
    print(f"PII substitutions - Emails: {pii_counts['email']}, Phones: {pii_counts['phone']}, Order IDs: {pii_counts['order_id']}")
    print(f"Total PII substitutions: {pii_counts['total']}")
    assert verify_no_pii_remaining(redacted_texts), "PII leakage check failed!"
    print("PII verification: 0 personal identifiers remain across all tickets.")

    # 2. Zero-Shot Evaluation Run
    print("\n--- 2. RUNNING ZERO-SHOT EVALUATION PIPELINE ---")
    zero_df, zero_eval = run_pipeline(
        tickets_path=tickets_path,
        segments_path=segments_path,
        mode="zero_shot",
        token_budget=100_000,
        reset_mock=True,
        log_retries=False,
    )
    zero_metrics = compute_metrics(zero_eval)
    print(f"Zero-shot classified: {zero_metrics['n_classified']} / {len(zero_df)}")
    print(f"Zero-shot accuracy: {zero_metrics['accuracy']:.4f} ({zero_metrics['correct']} / {zero_metrics['n_classified']})")
    print(f"Zero-shot total tokens: {zero_metrics['total_tokens']} (avg {zero_metrics['tokens_per_classified']:.2f} tokens/classified ticket)")
    print("Zero-shot confusion matrix:")
    print(zero_metrics["confusion_matrix"])

    # 3. Few-Shot Evaluation Run
    print("\n--- 3. RUNNING FEW-SHOT EVALUATION PIPELINE ---")
    few_df, few_eval = run_pipeline(
        tickets_path=tickets_path,
        segments_path=segments_path,
        mode="few_shot",
        token_budget=100_000,
        reset_mock=True,
        log_retries=False,
    )
    few_metrics = compute_metrics(few_eval)
    print(f"Few-shot classified: {few_metrics['n_classified']} / {len(few_df)}")
    print(f"Few-shot accuracy: {few_metrics['accuracy']:.4f} ({few_metrics['correct']} / {few_metrics['n_classified']})")
    print(f"Few-shot total tokens: {few_metrics['total_tokens']} (avg {few_metrics['tokens_per_classified']:.2f} tokens/classified ticket)")
    print("Few-shot confusion matrix:")
    print(few_metrics["confusion_matrix"])

    # 4. Token Budget Experiment (TOKEN_BUDGET = 6000)
    print("\n--- 4. EXECUTING TOKEN BUDGET DEMONSTRATION (BUDGET = 6000) ---")
    demo_df, _ = run_pipeline(
        tickets_path=tickets_path,
        segments_path=segments_path,
        mode="zero_shot",
        token_budget=6000,
        reset_mock=True,
        log_retries=False,
    )
    demo_df.to_csv(budget_demo_path, index=False)
    n_exhausted = (demo_df["status"] == "budget_exhausted").sum()
    print(f"Saved budget demo to: {budget_demo_path}")
    print(f"Tickets ending budget_exhausted: {n_exhausted}")

    # 5. Production Few-Shot Triage Run with Routing & Logging
    print("\n--- 5. EXECUTING PRODUCTION FEW-SHOT TRIAGE RUN (WITH LOGGING) ---")
    retry_records = []
    prod_triage_df, _ = run_pipeline(
        tickets_path=tickets_path,
        segments_path=segments_path,
        mode="few_shot",
        token_budget=100_000,
        reset_mock=True,
        log_retries=True,
        retry_records=retry_records,
    )
    prod_triage_df.to_csv(triage_results_path, index=False)
    print(f"Saved final triage results to: {triage_results_path}")
    print(f"Total retry occurrences handled: {len(retry_records)}")
    print("\nFinal Triage Routing Distribution:")
    print(prod_triage_df["route"].value_counts())
    print("\nFinal Triage Priority Distribution:")
    print(prod_triage_df["priority"].value_counts())

    # 6. Run Validation
    print("\n--- 6. RUNNING COMPREHENSIVE VALIDATION ---")
    validate_part_c(
        tickets_path=tickets_path,
        segments_path=segments_path,
        triage_results_path=triage_results_path,
        triage_budget_demo_path=budget_demo_path,
    )


if __name__ == "__main__":
    main()
