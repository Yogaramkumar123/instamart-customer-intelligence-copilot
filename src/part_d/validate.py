"""Comprehensive validation suite for Part D: Policy RAG Assistant and Refund Agent."""
import json
import math
import os
import sqlite3
from typing import Any, Dict, List
import pandas as pd

from src.part_d.document_loader import load_policy_documents
from src.part_d.chunking import chunk_policy_corpus
from src.part_d.retrieval import retrieve
from src.part_d.policy_assistant import evaluate_policy_rag, get_zero_overlap_vector_store
from src.part_d.tool_registry import TOOLS_SPEC, TOOL_MAP
from src.part_d.tools import (
    get_order,
    search_policy,
    calculate_refund,
    issue_refund,
    escalate_to_human,
)
from src.part_d.agent import run_agent, SimulatedAgentCrash
from src.part_d.resume import resume_agent


def test_refund_cap_and_idempotency(state_dir: str = "state") -> None:
    """Validate 10 direct safety and idempotency tests for refund operations."""
    test_ledger = os.path.join(state_dir, "test_security_ledger.jsonl")
    if os.path.exists(test_ledger):
        os.remove(test_ledger)

    # 1. ₹1,000 refund accepted
    r1 = issue_refund(
        order_id="ORD-SEC-01",
        refund_inr=1000.00,
        idempotency_key="key_test_1000",
        ledger_path=test_ledger,
    )
    assert "error" not in r1, f"Test 1 failed: expected 1000.00 accepted, got error: {r1.get('error')}"
    assert r1.get("status") == "issued"
    assert r1.get("refund_inr") == 1000.00

    with open(test_ledger, "r", encoding="utf-8") as f:
        lines_1 = [line.strip() for line in f if line.strip()]
    assert len(lines_1) == 1, f"Test 1 failed: ledger should have 1 record, got {len(lines_1)}"

    # 2. ₹1,000.01 refund rejected
    r2 = issue_refund(
        order_id="ORD-SEC-02",
        refund_inr=1000.01,
        idempotency_key="key_test_1000_01",
        ledger_path=test_ledger,
    )
    assert "error" in r2, "Test 2 failed: expected 1000.01 rejected, but no error returned"
    assert r2.get("status") == "rejected"

    # 3. ₹1,999 refund rejected
    r3 = issue_refund(
        order_id="ORD-SEC-03",
        refund_inr=1999.00,
        idempotency_key="key_test_1999",
        ledger_path=test_ledger,
    )
    assert "error" in r3, "Test 3 failed: expected 1999.00 rejected, but no error returned"
    assert r3.get("status") == "rejected"

    # 4. Rejected refund does not modify the ledger
    with open(test_ledger, "r", encoding="utf-8") as f:
        lines_4 = [line.strip() for line in f if line.strip()]
    assert len(lines_4) == 1, f"Test 4 failed: rejected refunds modified ledger! Expected 1, got {len(lines_4)}"

    # 5. Negative amount rejected
    r5 = issue_refund(
        order_id="ORD-SEC-05",
        refund_inr=-50.00,
        idempotency_key="key_test_neg",
        ledger_path=test_ledger,
    )
    assert "error" in r5, "Test 5 failed: expected negative amount rejected"
    assert r5.get("status") == "rejected"
    with open(test_ledger, "r", encoding="utf-8") as f:
        lines_5 = [line.strip() for line in f if line.strip()]
    assert len(lines_5) == 1, f"Test 5 failed: negative amount modified ledger! Expected 1, got {len(lines_5)}"

    # 6. NaN and infinity rejected
    r6_nan = issue_refund(
        order_id="ORD-SEC-06A",
        refund_inr=float("nan"),
        idempotency_key="key_test_nan",
        ledger_path=test_ledger,
    )
    assert "error" in r6_nan, "Test 6 failed: expected NaN rejected"
    assert r6_nan.get("status") == "rejected"

    r6_inf = issue_refund(
        order_id="ORD-SEC-06B",
        refund_inr=float("inf"),
        idempotency_key="key_test_inf",
        ledger_path=test_ledger,
    )
    assert "error" in r6_inf, "Test 6 failed: expected Infinity rejected"
    assert r6_inf.get("status") == "rejected"
    with open(test_ledger, "r", encoding="utf-8") as f:
        lines_6 = [line.strip() for line in f if line.strip()]
    assert len(lines_6) == 1, f"Test 6 failed: non-finite amount modified ledger! Expected 1, got {len(lines_6)}"

    # 7. Refund above ₹1,000 can be escalated
    r7 = escalate_to_human(
        order_id="ORD-SEC-07",
        refund_inr=1999.00,
        idempotency_key="key_test_esc_1999",
        ledger_path=test_ledger,
    )
    assert "error" not in r7, f"Test 7 failed: expected escalation above 1000 accepted, got error: {r7.get('error')}"
    assert r7.get("status") == "escalated"
    assert r7.get("refund_inr") == 1999.00
    with open(test_ledger, "r", encoding="utf-8") as f:
        lines_7 = [line.strip() for line in f if line.strip()]
    assert len(lines_7) == 2, f"Test 7 failed: escalation not appended to ledger! Expected 2, got {len(lines_7)}"

    # 8. Duplicate idempotency key does not issue a second refund
    r8_first = issue_refund(
        order_id="ORD-SEC-08",
        refund_inr=300.00,
        idempotency_key="key_test_duplicate_xyz",
        ledger_path=test_ledger,
    )
    assert "error" not in r8_first and r8_first.get("status") == "issued"
    with open(test_ledger, "r", encoding="utf-8") as f:
        count_before = len([l for l in f if l.strip()])

    r8_dup = issue_refund(
        order_id="ORD-SEC-08",
        refund_inr=300.00,
        idempotency_key="key_test_duplicate_xyz",
        ledger_path=test_ledger,
    )
    assert r8_dup.get("status") == "duplicate", f"Test 8 failed: expected duplicate status, got: {r8_dup}"
    assert "error" in r8_dup
    with open(test_ledger, "r", encoding="utf-8") as f:
        count_after = len([l for l in f if l.strip()])
    assert count_after == count_before, (
        f"Test 8 failed: duplicate key appended second record to ledger ({count_before} -> {count_after})"
    )

    # 9. Different idempotency keys can represent separate legitimate transactions
    r9_a = issue_refund(
        order_id="ORD-SEC-09",
        refund_inr=150.00,
        idempotency_key="key_test_separate_A",
        ledger_path=test_ledger,
    )
    assert "error" not in r9_a and r9_a.get("status") == "issued"

    r9_b = issue_refund(
        order_id="ORD-SEC-09",
        refund_inr=175.00,
        idempotency_key="key_test_separate_B",
        ledger_path=test_ledger,
    )
    assert "error" not in r9_b and r9_b.get("status") == "issued"

    with open(test_ledger, "r", encoding="utf-8") as f:
        entries_9 = [json.loads(line) for line in f if line.strip()]
    keys_9 = [e.get("idempotency_key") for e in entries_9]
    assert "key_test_separate_A" in keys_9 and "key_test_separate_B" in keys_9, (
        f"Test 9 failed: separate legitimate transaction keys missing: {keys_9}"
    )

    # 10. Crash/resume does not duplicate the same transaction
    crash_run_id = "test_security_crash_idem_run"
    try:
        run_agent(
            user_message="I need a refund for ORD-100048, delivery took 65 minutes",
            scenario_id="S1",
            run_id=crash_run_id,
            crash_after_tool_n=2,
            state_dir=state_dir,
            ledger_path=test_ledger,
        )
    except SimulatedAgentCrash:
        pass

    res_resumed = resume_agent(
        run_id=crash_run_id,
        state_dir=state_dir,
        ledger_path=test_ledger,
    )
    assert res_resumed["status"] == "REFUND"

    expected_idem_key = f"{crash_run_id}:ORD-100048"
    with open(test_ledger, "r", encoding="utf-8") as f:
        entries_10 = [json.loads(line) for line in f if line.strip()]
    matching_idem = [e for e in entries_10 if e.get("idempotency_key") == expected_idem_key]
    assert len(matching_idem) == 1, (
        f"Test 10 failed: expected exactly 1 transaction for crash/resume run, got {len(matching_idem)}"
    )

    re_exec = issue_refund(
        order_id="ORD-100048",
        refund_inr=128.25,
        idempotency_key=expected_idem_key,
        ledger_path=test_ledger,
    )
    assert re_exec.get("status") == "duplicate"
    with open(test_ledger, "r", encoding="utf-8") as f:
        entries_10_after = [json.loads(line) for line in f if line.strip()]
    matching_after = [e for e in entries_10_after if e.get("idempotency_key") == expected_idem_key]
    assert len(matching_after) == 1, (
        f"Test 10 failed: duplicate execution appended to ledger! Expected 1, got {len(matching_after)}"
    )

    if os.path.exists(test_ledger):
        os.remove(test_ledger)


def validate_part_d(
    policies_dir: str = "data/policies",
    policy_eval_csv: str = "data/raw/policy_eval.csv",
    scenarios_csv: str = "data/raw/agent_scenarios.csv",
    db_path: str = "data/instamart.db",
    runs_path: str = "outputs/agent_runs.jsonl",
    ledger_path: str = "outputs/refund_ledger.jsonl",
    state_dir: str = "state",
) -> None:
    """Validate all Part D functional requirements and numerical checkpoints."""
    print("=" * 60)
    print("RUNNING PART D COMPREHENSIVE VALIDATION SUITE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Document Loading and Chunking Parity (Task 1)
    # ---------------------------------------------------------
    docs = load_policy_documents(policies_dir)
    assert len(docs) == 5, f"Expected 5 policy documents, found {len(docs)}"

    expected_files = [
        "refund_policy.txt",
        "delivery_policy.txt",
        "payment_policy.txt",
        "privacy_policy.txt",
        "escalation_policy.txt",
    ]
    assert set(docs.keys()) == set(expected_files), f"Policy filenames mismatch: {list(docs.keys())}"

    chunks = chunk_policy_corpus(docs, chunk_size=40, chunk_overlap=10)
    assert len(chunks) == 15, f"Expected 15 chunks with overlap=10, got {len(chunks)}"

    chunk_counts = {}
    for c in chunks:
        chunk_counts[c["source"]] = chunk_counts.get(c["source"], 0) + 1

    assert chunk_counts.get("refund_policy.txt") == 4, f"refund_policy chunks expected 4, got {chunk_counts.get('refund_policy.txt')}"
    assert chunk_counts.get("delivery_policy.txt") == 3, f"delivery_policy chunks expected 3, got {chunk_counts.get('delivery_policy.txt')}"
    assert chunk_counts.get("escalation_policy.txt") == 3, f"escalation_policy chunks expected 3, got {chunk_counts.get('escalation_policy.txt')}"
    assert chunk_counts.get("payment_policy.txt") == 3, f"payment_policy chunks expected 3, got {chunk_counts.get('payment_policy.txt')}"
    assert chunk_counts.get("privacy_policy.txt") == 2, f"privacy_policy chunks expected 2, got {chunk_counts.get('privacy_policy.txt')}"

    print(f"  [PASS] Task 1: 5 policy files loaded, exactly 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2).")

    # ---------------------------------------------------------
    # 2. Vector Store and Retrieval (Task 2)
    # ---------------------------------------------------------
    retrieval_query = "refund policy late delivery"
    top_3 = retrieve(retrieval_query, k=3, policies_dir=policies_dir)
    assert len(top_3) == 3, f"Expected 3 chunks retrieved, got {len(top_3)}"

    for r in top_3:
        assert r["source"] == "refund_policy.txt", f"Expected refund_policy.txt chunk, got {r['source']}"
        assert r["score"] > 0.0, f"Expected positive cosine similarity score, got {r['score']}"

    print(f"  [PASS] Task 2: TF-IDF retrieval top 3 chunks for 'refund policy late delivery' all from refund_policy.txt with calculated cosine scores.")

    # ---------------------------------------------------------
    # 3. Grounding Evaluation on 14 Questions (Task 3)
    # ---------------------------------------------------------
    eval_10 = evaluate_policy_rag(eval_csv_path=policy_eval_csv)
    assert eval_10["answerable_count"] == 12, f"Expected 12 answerable questions, got {eval_10['answerable_count']}"
    assert eval_10["unanswerable_count"] == 2, f"Expected 2 unanswerable questions, got {eval_10['unanswerable_count']}"
    assert eval_10["doc_retrieved_count"] == 12, f"Expected 12/12 doc retrieved, got {eval_10['doc_retrieved_count']}"
    assert eval_10["fact_grounded_count"] == 12, f"Expected 12/12 fact grounded, got {eval_10['fact_grounded_count']}"
    assert eval_10["unanswerable_fallback_count"] == 2, f"Expected 2/2 unanswerable fallback, got {eval_10['unanswerable_fallback_count']}"

    print(f"  [PASS] Tasks 3 & 4: Grounding evaluation passed with 12/12 doc retrieved, 12/12 fact grounded, 2/2 unanswerable fallback.")

    # ---------------------------------------------------------
    # 4. Zero-Overlap Comparison Experiment (Task 4)
    # ---------------------------------------------------------
    store_zero = get_zero_overlap_vector_store(policies_dir=policies_dir)
    eval_zero = evaluate_policy_rag(eval_csv_path=policy_eval_csv, vector_store=store_zero)
    assert len(eval_zero["failures"]) >= 1, f"Expected at least 1 failure under zero-overlap, got {len(eval_zero['failures'])}"

    print(f"  [PASS] Task 4 Zero-Overlap: Successfully reproduced failure in zero-overlap experiment ({len(eval_zero['failures'])} questions failed due to chunk boundary splitting).")

    # ---------------------------------------------------------
    # 5. Tool Specifications & Dispatch Registry (Task 5)
    # ---------------------------------------------------------
    expected_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund", "escalate_to_human"]
    assert len(TOOLS_SPEC) == 5, f"Expected 5 tool specs, got {len(TOOLS_SPEC)}"
    spec_names = [t["name"] for t in TOOLS_SPEC]
    assert set(spec_names) == set(expected_tools), f"Tool specs mismatch: {spec_names}"

    assert len(TOOL_MAP) == 5, f"Expected 5 tools in TOOL_MAP, got {len(TOOL_MAP)}"
    assert set(TOOL_MAP.keys()) == set(expected_tools), f"TOOL_MAP mismatch: {list(TOOL_MAP.keys())}"

    # Verify tool functions
    o_good = get_order("ORD-100048", db_path=db_path)
    assert o_good.get("order_id") == "ORD-100048"
    assert "status" in o_good and "order_amount" in o_good and "delivery_minutes" in o_good

    o_bad = get_order("ORD-999999", db_path=db_path)
    assert o_bad.get("error") == "order not found"

    calc_late = calculate_refund(order_amount=200.0, delivery_minutes=50)
    assert calc_late.get("refund_inr") == 200.0

    calc_ontime = calculate_refund(order_amount=200.0, delivery_minutes=30)
    assert calc_ontime.get("refund_inr") == 0.0

    print(f"  [PASS] Task 5: Tool specs, dispatch map, DB lookups, and refund calculation rules verified.")

    # ---------------------------------------------------------
    # 6. Direct Financial Safety & Idempotency Tests (Tasks 1, 2 & 3)
    # ---------------------------------------------------------
    test_refund_cap_and_idempotency(state_dir=state_dir)
    print(f"  [PASS] Financial Safety & Idempotency: All 10 direct safety tests verified (INR 1000 cap, rejected amounts, NaN/Inf, duplicate keys, separate transactions, crash idempotency).")

    # ---------------------------------------------------------
    # 7. ReAct Agent Loop Across Seven Scenarios (Task 6)
    # ---------------------------------------------------------
    scenarios_df = pd.read_csv(scenarios_csv)
    assert len(scenarios_df) == 7, f"Expected 7 scenarios, got {len(scenarios_df)}"

    val_ledger_path = os.path.join(state_dir, "validation_ledger.jsonl")
    if os.path.exists(val_ledger_path):
        os.remove(val_ledger_path)

    scenario_results = {}
    for _, row in scenarios_df.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        exp_out = row["expected_outcome"]
        exp_amt = float(row["expected_refund_inr"])

        res = run_agent(
            user_message=msg,
            scenario_id=sid,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            ledger_path=val_ledger_path,
        )
        scenario_results[sid] = res

        if sid == "S7":
            assert res["status"] == "FALLBACK_MAX_ITERATIONS", f"S7 expected FALLBACK_MAX_ITERATIONS, got {res['status']}"
            assert res["llm_calls"] == 6, f"S7 expected 6 model calls, got {res['llm_calls']}"
        else:
            assert res["status"] == exp_out, f"Scenario {sid} expected {exp_out}, got {res['status']}"
            assert abs(res["refund_amount"] - exp_amt) < 0.01, f"Expected refund {exp_amt}, got {res['refund_amount']}"

    assert scenario_results["S1"]["llm_calls"] == 5, f"S1 expected 5 model calls, got {scenario_results['S1']['llm_calls']}"
    assert scenario_results["S1"]["status"] == "REFUND"
    assert scenario_results["S2"]["status"] == "REFUND"
    assert scenario_results["S3"]["status"] == "ESCALATE"
    assert scenario_results["S4"]["status"] == "DENIED"
    assert scenario_results["S5"]["status"] == "DENIED"
    assert scenario_results["S6"]["status"] == "NOT_FOUND"

    print(f"  [PASS] Task 6: All seven scenarios matched expected outcomes, S1 took 5 calls, S7 reached FALLBACK_MAX_ITERATIONS.")

    # ---------------------------------------------------------
    # 8. Safe Fallback Limits (Task 7)
    # ---------------------------------------------------------
    s1_msg = scenarios_df[scenarios_df["scenario_id"] == "S1"]["message"].iloc[0]
    res_tok = run_agent(s1_msg, "S1_tok_demo", max_iterations=6, token_limit=200, time_limit_seconds=10, ledger_path=val_ledger_path)
    assert res_tok["status"] == "FALLBACK_TOKEN_LIMIT", f"Expected FALLBACK_TOKEN_LIMIT, got {res_tok['status']}"

    res_time = run_agent(s1_msg, "S1_time_demo", max_iterations=6, token_limit=6000, time_limit_seconds=0, ledger_path=val_ledger_path)
    assert res_time["status"] == "FALLBACK_TIMEOUT", f"Expected FALLBACK_TIMEOUT, got {res_time['status']}"

    print(f"  [PASS] Task 7: Token limit demo (FALLBACK_TOKEN_LIMIT) and Timeout demo (FALLBACK_TIMEOUT) verified.")

    # ---------------------------------------------------------
    # 9. Persistent Checkpoint & Crash Recovery (Task 8)
    # ---------------------------------------------------------
    crash_run_id = "val_crash_demo_s1"
    crash_occurred = False
    try:
        run_agent(
            user_message=s1_msg,
            scenario_id="S1",
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            run_id=crash_run_id,
            crash_after_tool_n=2,
            state_dir=state_dir,
            ledger_path=val_ledger_path,
        )
    except SimulatedAgentCrash:
        crash_occurred = True

    assert crash_occurred, "Simulated crash failed to trigger!"

    resumed_res = resume_agent(
        run_id=crash_run_id,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10,
        state_dir=state_dir,
        ledger_path=val_ledger_path,
    )

    assert resumed_res["status"] == "REFUND", f"Resumed status expected REFUND, got {resumed_res['status']}"
    assert "128.25" in resumed_res["final_answer"], f"Expected 128.25 in answer: {resumed_res['final_answer']}"
    assert resumed_res["tools_executed"] == [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ], f"Tools executed sequence mismatch: {resumed_res['tools_executed']}"

    print(f"  [PASS] Task 8: Crash and resume demonstration verified (exact 4 tool sequence: get_order, search_policy, calculate_refund, issue_refund; status REFUND INR 128.25).")

    # ---------------------------------------------------------
    # 10. Output Files & Artifact Validation
    # ---------------------------------------------------------
    assert os.path.exists(ledger_path), f"Missing ledger file: {ledger_path}"
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger_lines = [json.loads(line) for line in f if line.strip()]
    assert len(ledger_lines) > 0, "Refund ledger is empty!"

    tools_in_ledger = set(l["tool"] for l in ledger_lines)
    assert "issue_refund" in tools_in_ledger, "issue_refund actions missing from ledger"
    assert "escalate_to_human" in tools_in_ledger, "escalate_to_human actions missing from ledger"

    assert os.path.exists(runs_path), f"Missing runs file: {runs_path}"
    with open(runs_path, "r", encoding="utf-8") as f:
        runs_lines = [json.loads(line) for line in f if line.strip()]
    assert len(runs_lines) == 7, f"Expected 7 records in {runs_path}, got {len(runs_lines)}"

    print(f"  [PASS] Artifacts: Outputs ledger validated with {len(ledger_lines)} records (both issue_refund and escalate_to_human present).")
    print("=" * 60)
    print("ALL PART D CHECKPOINTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    validate_part_d()
