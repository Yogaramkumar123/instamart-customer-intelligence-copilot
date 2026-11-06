"""Run full Part D: Policy RAG Assistant and Refund Agent pipeline."""
import json
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.part_d.document_loader import load_policy_documents
from src.part_d.chunking import chunk_policy_corpus
from src.part_d.retrieval import retrieve, get_default_vector_store, get_zero_overlap_vector_store
from src.part_d.policy_assistant import evaluate_policy_rag
from src.part_d.agent import run_agent, SimulatedAgentCrash
from src.part_d.resume import resume_agent
from src.part_d.validate import validate_part_d


def main() -> None:
    print("=" * 70)
    print("STARTING PART D: POLICY RAG ASSISTANT AND REFUND AGENT PIPELINE")
    print("=" * 70)

    policies_dir = "data/policies"
    policy_eval_csv = "data/raw/policy_eval.csv"
    scenarios_csv = "data/raw/agent_scenarios.csv"
    runs_path = "outputs/agent_runs.jsonl"
    ledger_path = "outputs/refund_ledger.jsonl"
    state_dir = "state"

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    # Reset ledger and runs file for clean reproducible execution
    if os.path.exists(runs_path):
        os.remove(runs_path)
    if os.path.exists(ledger_path):
        os.remove(ledger_path)

    # 1. Document Loading and Chunking
    print("\n--- 1. LOADING AND CHUNKING POLICY DOCUMENTS ---")
    docs = load_policy_documents(policies_dir)
    chunks_10 = chunk_policy_corpus(docs, chunk_size=40, chunk_overlap=10)
    print(f"Loaded {len(docs)} policy documents.")
    chunk_counts = {}
    for c in chunks_10:
        chunk_counts[c["source"]] = chunk_counts.get(c["source"], 0) + 1
    print("Chunk counts per document (chunk_size=40, chunk_overlap=10):")
    for doc, cnt in chunk_counts.items():
        print(f"  - {doc}: {cnt} chunks")
    print(f"Total chunks: {len(chunks_10)}")

    # 2. Vectorization & Retrieval Demonstration
    print("\n--- 2. TF-IDF VECTORIZATION AND RETRIEVAL ---")
    query = "refund policy late delivery"
    retrieved = retrieve(query, k=3, policies_dir=policies_dir)
    print(f"Query: '{query}' -> Top 3 chunks:")
    for i, r in enumerate(retrieved, 1):
        print(f"  [{i}] Source: {r['source']} | Score: {r['score']:.4f} | Text: {r['text'][:55]}...")

    # 3. Policy Grounding Evaluation (Overlap = 10)
    print("\n--- 3. EVALUATING POLICY RAG GROUNDING (OVERLAP = 10) ---")
    res_10 = evaluate_policy_rag(eval_csv_path=policy_eval_csv)
    print(f"Answerable Questions Evaluated: {res_10['answerable_count']}")
    print(f"  - Expected Document Retrieved in Top 3: {res_10['doc_retrieved_count']} / {res_10['answerable_count']}")
    print(f"  - Expected Fact Found in Answer: {res_10['fact_grounded_count']} / {res_10['answerable_count']}")
    print(f"Unanswerable Questions Evaluated: {res_10['unanswerable_count']}")
    print(f"  - Correct Fallback ('I could not find...'): {res_10['unanswerable_fallback_count']} / {res_10['unanswerable_count']}")

    # 4. Zero-Overlap Comparison Experiment (Overlap = 0)
    print("\n--- 4. EVALUATING ZERO-OVERLAP RAG (OVERLAP = 0) ---")
    store_0 = get_zero_overlap_vector_store(policies_dir=policies_dir)
    res_0 = evaluate_policy_rag(eval_csv_path=policy_eval_csv, vector_store=store_0)
    print(f"Zero-Overlap Answerable Doc in Top 3: {res_0['doc_retrieved_count']} / {res_0['answerable_count']}")
    print(f"Zero-Overlap Answerable Fact Grounded: {res_0['fact_grounded_count']} / {res_0['answerable_count']}")
    print(f"Zero-Overlap Failures Detected: {len(res_0['failures'])}")
    for f in res_0["failures"]:
        print(f"  * Failed Q{f['q_num']}: '{f['question']}'")
        print(f"    Reason: {f['reason']}")

    # 5. Seven ReAct Agent Scenarios
    print("\n--- 5. RUNNING SEVEN AGENT SCENARIOS (REACT LOOP) ---")
    scenarios_df = pd.read_csv(scenarios_csv)
    scenario_runs = []

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
            ledger_path=ledger_path,
            state_dir=state_dir,
        )

        scenario_runs.append({
            "scenario_id": sid,
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"],
            "tools_executed": res["tools_executed"],
        })

        print(f"Scenario {sid}: status={res['status']:<23} | calls={res['llm_calls']} | tokens={res['tokens_used']} | tools={res['tools_executed']}")
        print(f"  Answer: {res['final_answer']}")

    # Persist agent_runs.jsonl
    with open(runs_path, "w", encoding="utf-8") as f:
        for r in scenario_runs:
            f.write(json.dumps(r) + "\n")
    print(f"\nPersisted {len(scenario_runs)} scenario records to {runs_path}")

    # 6. Safety Limits Demonstrations
    print("\n--- 6. SAFETY LIMITS & FALLBACK DEMONSTRATIONS ---")
    s1_msg = scenarios_df[scenarios_df["scenario_id"] == "S1"]["message"].iloc[0]

    demo_token = run_agent(s1_msg, "S1_demo_token", max_iterations=6, token_limit=200, time_limit_seconds=10, ledger_path=ledger_path)
    print(f"Token Ceiling Demo (limit=200): status={demo_token['status']}")

    demo_timeout = run_agent(s1_msg, "S1_demo_timeout", max_iterations=6, token_limit=6000, time_limit_seconds=0, ledger_path=ledger_path)
    print(f"Timeout Demo (time_limit=0s): status={demo_timeout['status']}")

    # 7. Persistent Crash & Resume Demonstration on S1
    print("\n--- 7. PERSISTENT CRASH AND RESUME DEMONSTRATION ON S1 ---")
    crash_run_id = "crash_recovery_demo_s1"
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
            ledger_path=ledger_path,
        )
    except SimulatedAgentCrash as err:
        print(f"Phase 1: {err}")

    resumed_res = resume_agent(
        run_id=crash_run_id,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10,
        state_dir=state_dir,
        ledger_path=ledger_path,
    )
    print(f"Phase 2: Resumed execution completed with status={resumed_res['status']}")
    print(f"Final Answer: {resumed_res['final_answer']}")
    print(f"Tools executed across crash and resume: {resumed_res['tools_executed']}")

    # 8. Run Automated Validation Suite
    print("\n--- 8. EXECUTING AUTOMATED VALIDATION SUITE ---")
    validate_part_d(
        policies_dir=policies_dir,
        policy_eval_csv=policy_eval_csv,
        scenarios_csv=scenarios_csv,
        db_path="data/instamart.db",
        ledger_path=ledger_path,
        runs_path=runs_path,
        state_dir=state_dir,
    )


if __name__ == "__main__":
    main()
