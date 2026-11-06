"""Runtime validation suite for Part C: Support Ticket Triage with Guardrails.

Asserts all reference checkpoints, guardrails, PII redactions, prompt structures,
retries, classification outputs, accuracies, budget demo, and routing rules.
Prints exactly 'ALL PART C CHECKPOINTS PASSED' when all hold.
"""
import os
import re
import pandas as pd
import mock_llm

from src.part_c.guardrails import check_input_guardrails, MAX_INPUT_TOKENS
from src.part_c.pii_redaction import (
    audit_corpus_pii,
    verify_no_pii_remaining,
    EMAIL_PATTERN,
    PHONE_PATTERN,
    ORDER_ID_PATTERN,
)
from src.part_c.prompts import (
    ZERO_SHOT_SYSTEM_PROMPT,
    FEW_SHOT_SYSTEM_PROMPT,
    ALLOWED_CATEGORIES,
)
from src.part_c.pipeline import run_pipeline, compute_metrics


def validate_part_c(
    tickets_path: str = "data/raw/tickets.csv",
    segments_path: str = "outputs/customer_segments.csv",
    triage_results_path: str = "outputs/triage_results.csv",
    triage_budget_demo_path: str = "outputs/triage_budget_demo.csv",
) -> None:
    """Execute all Part C runtime verification checks."""
    print("=" * 60)
    print("RUNNING PART C VERIFICATION SUITE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Input Data & Guardrails (Task 1)
    # ---------------------------------------------------------
    assert os.path.exists(tickets_path), f"Missing {tickets_path}"
    tickets_df = pd.read_csv(tickets_path, keep_default_na=False)
    assert len(tickets_df) == 164, f"Expected 164 tickets, got {len(tickets_df)}"

    empty_tickets = []
    long_tickets = []
    for _, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        txt = row["text"]
        g_status = check_input_guardrails(txt)
        if g_status == "skipped_empty":
            empty_tickets.append(tid)
        elif g_status == "flagged_long":
            tokens = mock_llm.est_tokens(txt)
            assert tokens == 734, f"Expected 734 tokens for long ticket {tid}, got {tokens}"
            long_tickets.append(tid)

    assert len(empty_tickets) == 4, f"Expected 4 empty tickets, got {len(empty_tickets)}"
    assert len(long_tickets) == 4, f"Expected 4 long tickets, got {len(long_tickets)}"
    print(f"[CHECK 1] Guardrails verified: 4 empty tickets, 4 long tickets (each 734 tokens).")

    # ---------------------------------------------------------
    # 2. PII Redaction Audit (Task 2)
    # ---------------------------------------------------------
    raw_texts = tickets_df["text"].tolist()
    redacted_texts, pii_counts = audit_corpus_pii(raw_texts)

    assert pii_counts["email"] == 55, f"Expected 55 emails redacted, got {pii_counts['email']}"
    assert pii_counts["phone"] == 91, f"Expected 91 phones redacted, got {pii_counts['phone']}"
    assert pii_counts["order_id"] == 132, f"Expected 132 orders redacted, got {pii_counts['order_id']}"
    assert pii_counts["total"] == 278, f"Expected 278 total redacted, got {pii_counts['total']}"
    assert verify_no_pii_remaining(redacted_texts), "PII remnants detected after redaction!"
    print(f"[CHECK 2] PII redaction verified: 55 emails, 91 phones, 132 orders (278 total, 0 remaining).")

    # ---------------------------------------------------------
    # 3. Prompts Verification (Task 3)
    # ---------------------------------------------------------
    # Zero-shot checks
    assert "CLASSIFY" in ZERO_SHOT_SYSTEM_PROMPT, "Zero-shot prompt missing 'CLASSIFY'"
    for cat in ALLOWED_CATEGORIES:
        assert cat in ZERO_SHOT_SYSTEM_PROMPT, f"Zero-shot prompt missing category '{cat}'"
    assert '{"category": "...", "reason": "..."}' in ZERO_SHOT_SYSTEM_PROMPT, "Zero-shot missing JSON schema"
    assert "EXAMPLES:" not in ZERO_SHOT_SYSTEM_PROMPT, "Zero-shot prompt must NOT contain 'EXAMPLES:'"

    # Few-shot checks
    assert "CLASSIFY" in FEW_SHOT_SYSTEM_PROMPT, "Few-shot prompt missing 'CLASSIFY'"
    for cat in ALLOWED_CATEGORIES:
        assert cat in FEW_SHOT_SYSTEM_PROMPT, f"Few-shot prompt missing category '{cat}'"
    assert '{"category": "...", "reason": "..."}' in FEW_SHOT_SYSTEM_PROMPT, "Few-shot missing JSON schema"
    assert "EXAMPLES:" in FEW_SHOT_SYSTEM_PROMPT, "Few-shot prompt must contain 'EXAMPLES:'"
    assert '"category": "fraud"' in FEW_SHOT_SYSTEM_PROMPT, "Few-shot prompt missing fraud example"
    print(f"[CHECK 3] System prompts verified: Zero-shot & Few-shot contract compliant.")

    # ---------------------------------------------------------
    # 4. Resilient Calls & Retries (Task 4)
    # ---------------------------------------------------------
    retry_records = []
    run_pipeline(
        tickets_path=tickets_path,
        segments_path=segments_path,
        mode="few_shot",
        token_budget=100_000,
        reset_mock=True,
        log_retries=False,
        retry_records=retry_records,
    )
    assert len(retry_records) == 19, f"Expected exactly 19 retries, got {len(retry_records)}"
    for rec in retry_records:
        assert rec["retry_n"] == 0, f"Expected retry number 0, got {rec['retry_n']}"
        assert 0.045 <= rec["wait"] <= 0.0575, f"Wait time {rec['wait']} outside [0.045, 0.0575]s"
    print(f"[CHECK 4] Retry resiliency verified: exactly 19 TransientLLMErrors recovered on retry 0 within [0.045, 0.0575]s.")

    # ---------------------------------------------------------
    # 5. Output Validation Counts (Task 5)
    # ---------------------------------------------------------
    zero_df, zero_eval = run_pipeline(
        tickets_path=tickets_path,
        segments_path=segments_path,
        mode="zero_shot",
        token_budget=100_000,
        reset_mock=True,
        log_retries=False,
    )
    status_counts = zero_df["status"].value_counts().to_dict()
    assert status_counts.get("classified_json", 0) == 135, f"Expected 135 classified_json, got {status_counts.get('classified_json')}"
    assert status_counts.get("classified_keyword", 0) == 12, f"Expected 12 classified_keyword, got {status_counts.get('classified_keyword')}"
    assert status_counts.get("flagged_invalid_output", 0) == 9, f"Expected 9 flagged_invalid_output, got {status_counts.get('flagged_invalid_output')}"
    assert status_counts.get("skipped_empty", 0) == 4, f"Expected 4 skipped_empty, got {status_counts.get('skipped_empty')}"
    assert status_counts.get("flagged_long", 0) == 4, f"Expected 4 flagged_long, got {status_counts.get('flagged_long')}"
    print(f"[CHECK 5] Output validation counts verified: 135 JSON, 12 keyword, 9 invalid output (147 classified).")

    # ---------------------------------------------------------
    # 6. Zero-Shot vs Few-Shot Accuracies (Task 6)
    # ---------------------------------------------------------
    zero_metrics = compute_metrics(zero_eval)
    assert zero_metrics["n_classified"] == 147, f"Expected 147 classified in zero-shot, got {zero_metrics['n_classified']}"
    assert zero_metrics["correct"] == 126, f"Expected 126 correct in zero-shot, got {zero_metrics['correct']}"
    assert abs(zero_metrics["accuracy"] - (126 / 147)) < 0.005, f"Zero-shot accuracy mismatch: {zero_metrics['accuracy']}"

    # Verify zero-shot error distribution: 11 fraud->billing, 10 technical->billing
    err_dict = {}
    for tid, tcat, pcat in zero_metrics["errors"]:
        key = f"{tcat}->{pcat}"
        err_dict[key] = err_dict.get(key, 0) + 1
    assert err_dict.get("fraud->billing", 0) == 11, f"Expected 11 fraud->billing errors, got {err_dict.get('fraud->billing')}"
    assert err_dict.get("technical->billing", 0) == 10, f"Expected 10 technical->billing errors, got {err_dict.get('technical->billing')}"
    assert len(zero_metrics["errors"]) == 21, f"Expected 21 total errors, got {len(zero_metrics['errors'])}"

    few_df, few_eval = run_pipeline(
        tickets_path=tickets_path,
        segments_path=segments_path,
        mode="few_shot",
        token_budget=100_000,
        reset_mock=True,
        log_retries=False,
    )
    few_metrics = compute_metrics(few_eval)
    assert few_metrics["n_classified"] == 147, f"Expected 147 classified in few-shot, got {few_metrics['n_classified']}"
    assert few_metrics["correct"] == 147, f"Expected 147 correct in few-shot, got {few_metrics['correct']}"
    assert abs(few_metrics["accuracy"] - 1.0) < 0.001, f"Few-shot accuracy mismatch: {few_metrics['accuracy']}"
    print(f"[CHECK 6] Classification accuracies verified: Zero-shot 85.7% (126/147, 11 fraud & 10 tech -> billing), Few-shot 100.0% (147/147).")

    # ---------------------------------------------------------
    # 7. Token Budget Demo (Task 7a)
    # ---------------------------------------------------------
    assert os.path.exists(triage_budget_demo_path), f"Missing {triage_budget_demo_path}"
    demo_df = pd.read_csv(triage_budget_demo_path)
    assert len(demo_df) == 164, f"Expected 164 rows in demo, got {len(demo_df)}"
    n_exhausted = (demo_df["status"] == "budget_exhausted").sum()
    assert n_exhausted >= 1, f"Expected at least 1 budget_exhausted ticket, got {n_exhausted}"
    print(f"[CHECK 7] Token budget demo verified: {n_exhausted} tickets ended budget_exhausted under 6000 token ceiling.")

    # ---------------------------------------------------------
    # 8. Final Triage Results & Routing Rules (Task 7b)
    # ---------------------------------------------------------
    assert os.path.exists(triage_results_path), f"Missing {triage_results_path}"
    triage_df = pd.read_csv(triage_results_path, keep_default_na=False)
    assert len(triage_df) == 164, f"Expected 164 rows in triage results, got {len(triage_df)}"
    expected_cols = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]
    assert list(triage_df.columns) == expected_cols, f"Columns mismatch: {list(triage_df.columns)}"

    # Validate routing consistency
    for _, r in triage_df.iterrows():
        st = r["status"]
        cat = r["category"]
        prio = r["priority"]
        route = r["route"]
        cid = r["customer_id"]

        if st in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            assert route == "human_review", f"Unclassified ticket {r['ticket_id']} routed to {route}"
            assert prio == "NORMAL", f"Unclassified ticket {r['ticket_id']} priority {prio}"
            assert cat == "", f"Unclassified ticket {r['ticket_id']} had category '{cat}'"
        elif cat == "fraud":
            assert route == "security_team", f"Fraud ticket {r['ticket_id']} routed to {route}"
            assert prio == "HIGH", f"Fraud ticket {r['ticket_id']} priority {prio}"
        else:
            # Non-fraud classified
            # Look up premium status in segments
            # In our lookup, if is_premium -> premium_desk, HIGH; else -> support_queue, NORMAL
            if route == "premium_desk":
                assert prio == "HIGH", f"Premium desk ticket {r['ticket_id']} priority {prio}"
            elif route == "support_queue":
                assert prio == "NORMAL", f"Support queue ticket {r['ticket_id']} priority {prio}"
            else:
                raise AssertionError(f"Unexpected route {route} for ticket {r['ticket_id']}")

    print(f"[CHECK 8] Routing & priorities verified: human_review (NORMAL), security_team (HIGH), premium_desk (HIGH), support_queue (NORMAL).")
    print("=" * 60)
    print("ALL PART C CHECKPOINTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    validate_part_c()
