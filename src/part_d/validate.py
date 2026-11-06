"""Automated runtime validation suite for Part D: Policy RAG Assistant and Refund Agent.

Asserts all requirements across document loading, chunking, TF-IDF vector retrieval,
policy RAG assistant, grounding evaluation, zero-overlap comparison, tool specifications,
agent ReAct loop across 7 scenarios, safe fallback limits, persistent checkpoint recovery,
and ledger logging.

Prints:
==================================================
ALL PART D CHECKPOINTS PASSED
==================================================
"""
import json
import os
import sqlite3
import pandas as pd

from src.part_d.document_loader import load_policy_documents, POLICY_FILES
from src.part_d.chunking import chunk_policy_corpus, EXPECTED_CHUNK_COUNTS
from src.part_d.retrieval import (
    retrieve,
    get_default_vector_store,
    get_zero_overlap_vector_store,
)
from src.part_d.policy_assistant import answer, evaluate_policy_rag
from src.part_d.tools import (
    get_order,
    search_policy,
    calculate_refund,
    issue_refund,
    escalate_to_human,
)
from src.part_d.tool_registry import TOOL_MAP, TOOLS_SPEC
from src.part_d.refund_calculator import calculate_refund_amount
from src.part_d.agent import run_agent, SimulatedAgentCrash
from src.part_d.resume import resume_agent


def validate_part_d(
    policies_dir: str = "data/policies",
    policy_eval_csv: str = "data/raw/policy_eval.csv",
    scenarios_csv: str = "data/raw/agent_scenarios.csv",
    db_path: str = "data/instamart.db",
    ledger_path: str = "outputs/refund_ledger.jsonl",
    runs_path: str = "outputs/agent_runs.jsonl",
    state_dir: str = "state",
) -> None:
    """Execute all Part D runtime verification checks."""
    print("=" * 60)
    print("RUNNING PART D COMPREHENSIVE VALIDATION SUITE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Document Loading and Chunking Validation (Task 1)
    # ---------------------------------------------------------
    docs = load_policy_documents(policies_dir)
    assert len(docs) == 5, f"Expected 5 policy documents, found {len(docs)}"
    for fn in POLICY_FILES:
        assert fn in docs, f"Missing document {fn}"

    chunks_10 = chunk_policy_corpus(docs, chunk_size=40, chunk_overlap=10)
    assert len(chunks_10) == 15, f"Expected 15 total chunks, got {len(chunks_10)}"

    chunk_counts = {}
    for c in chunks_10:
        chunk_counts[c["source"]] = chunk_counts.get(c["source"], 0) + 1
        assert c["word_count"] <= 40, f"Chunk exceeded 40 words: {c['word_count']}"

    for fn, expected_count in EXPECTED_CHUNK_COUNTS.items():
        actual = chunk_counts.get(fn, 0)
        assert actual == expected_count, f"Expected {expected_count} chunks for {fn}, got {actual}"

    print(f"  [PASS] Task 1: 5 policy files loaded, exactly 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2).")

    # ---------------------------------------------------------
    # 2. TF-IDF Vectorization and Retrieval (Task 2)
    # ---------------------------------------------------------
    res = retrieve("refund policy late delivery", k=3, policies_dir=policies_dir)
    assert len(res) == 3, f"Expected 3 retrieved chunks, got {len(res)}"
    for r in res:
        assert "source" in r and "score" in r and "text" in r
        assert isinstance(r["score"], float) and r["score"] > 0
        assert r["source"] == "refund_policy.txt", f"Expected refund_policy.txt, got {r['source']}"

    print(f"  [PASS] Task 2: TF-IDF retrieval top 3 chunks for 'refund policy late delivery' all from refund_policy.txt with calculated cosine scores.")

    # ---------------------------------------------------------
    # 3. Policy RAG Assistant & Grounding Evaluation (Tasks 3 & 4)
    # ---------------------------------------------------------
    eval_overlap = evaluate_policy_rag(eval_csv_path=policy_eval_csv)
    assert eval_overlap["doc_retrieved_count"] == 12, f"Expected 12/12 docs retrieved, got {eval_overlap['doc_retrieved_count']}"
    assert eval_overlap["fact_grounded_count"] == 12, f"Expected 12/12 facts grounded, got {eval_overlap['fact_grounded_count']}"
    assert eval_overlap["unanswerable_fallback_count"] == 2, f"Expected 2/2 fallbacks, got {eval_overlap['unanswerable_fallback_count']}"
    assert len(eval_overlap["failures"]) == 0, f"Unexpected failures in overlap evaluation: {eval_overlap['failures']}"

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
    # 6. ReAct Agent Loop Across Seven Scenarios (Task 6)
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
    # 7. Safe Fallback Limits (Task 7)
    # ---------------------------------------------------------
    s1_msg = scenarios_df[scenarios_df["scenario_id"] == "S1"]["message"].iloc[0]
    res_tok = run_agent(s1_msg, "S1_tok_demo", max_iterations=6, token_limit=200, time_limit_seconds=10, ledger_path=val_ledger_path)
    assert res_tok["status"] == "FALLBACK_TOKEN_LIMIT", f"Expected FALLBACK_TOKEN_LIMIT, got {res_tok['status']}"

    res_time = run_agent(s1_msg, "S1_time_demo", max_iterations=6, token_limit=6000, time_limit_seconds=0, ledger_path=val_ledger_path)
    assert res_time["status"] == "FALLBACK_TIMEOUT", f"Expected FALLBACK_TIMEOUT, got {res_time['status']}"

    print(f"  [PASS] Task 7: Token limit demo (FALLBACK_TOKEN_LIMIT) and Timeout demo (FALLBACK_TIMEOUT) verified.")

    # ---------------------------------------------------------
    # 8. Persistent Checkpoint & Crash Recovery (Task 8)
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
    # 9. Output Files & Artifact Validation
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
