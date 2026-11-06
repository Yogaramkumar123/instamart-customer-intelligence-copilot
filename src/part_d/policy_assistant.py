"""Policy Question-Answering Assistant (RAG) and Evaluation Suite."""
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import mock_llm
from src.part_d.retrieval import retrieve, get_default_vector_store, get_zero_overlap_vector_store
from src.part_d.vector_store import PolicyVectorStore


POLICY_ASSISTANT_SYSTEM_PROMPT = (
    "You are an automated POLICY ASSISTANT. Answer the customer's question strictly "
    "using the provided policy context. Do not invent or extrapolate policies. "
    "If the answer cannot be found in the context, state that you could not find that information."
)


def answer(
    question: str,
    vector_store: Optional[PolicyVectorStore] = None,
    k: int = 3,
) -> Tuple[str, List[str]]:
    """Answer a customer policy question using retrieval-augmented generation.
    
    Returns:
        (answer_text, list_of_source_filenames)
    """
    if vector_store is not None:
        chunks = vector_store.retrieve(question, k=k)
    else:
        chunks = retrieve(question, k=k)

    context_lines = [f"[{c['source']}] {c['text']}" for c in chunks]
    context_str = "\n".join(context_lines)

    user_message_content = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"

    messages = [
        {"role": "system", "content": POLICY_ASSISTANT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message_content},
    ]

    response = mock_llm.chat(messages)
    answer_text = response.choices[0].message.content
    sources = [c["source"] for c in chunks]

    return answer_text, sources


def evaluate_policy_rag(
    eval_csv_path: str = "data/raw/policy_eval.csv",
    vector_store: Optional[PolicyVectorStore] = None,
) -> Dict[str, Any]:
    """Run full grounding and retrieval evaluation across the 14 policy questions.
    
    Returns metrics summary dictionary with itemized question outcomes.
    """
    df = pd.read_csv(eval_csv_path)

    total_questions = len(df)
    answerable_count = 0
    doc_retrieved_count = 0
    fact_grounded_count = 0
    unanswerable_count = 0
    unanswerable_fallback_count = 0

    detailed_results = []
    failures = []

    for idx, row in df.iterrows():
        q_num = idx + 1
        q = str(row["question"])
        exp_doc = str(row["expected_doc"])
        exp_fact = str(row["expected_fact"]) if pd.notna(row["expected_fact"]) else ""

        ans, sources = answer(q, vector_store=vector_store, k=3)

        if exp_doc == "NONE":
            unanswerable_count += 1
            is_fallback = ans.startswith("I could not find")
            if is_fallback:
                unanswerable_fallback_count += 1
            else:
                failures.append({
                    "q_num": q_num,
                    "question": q,
                    "reason": f"Expected fallback starting with 'I could not find', got '{ans}'",
                })
            detailed_results.append({
                "q_num": q_num,
                "question": q,
                "expected_doc": "NONE",
                "expected_fact": "",
                "sources": sources,
                "answer": ans,
                "doc_retrieved": True,
                "fact_grounded": is_fallback,
                "status": "PASS" if is_fallback else "FAIL",
            })
        else:
            answerable_count += 1
            doc_ok = exp_doc in sources
            fact_ok = exp_fact in ans

            if doc_ok:
                doc_retrieved_count += 1
            if fact_ok:
                fact_grounded_count += 1

            passed = doc_ok and fact_ok
            if not passed:
                reasons = []
                if not doc_ok:
                    reasons.append(f"Expected doc {exp_doc} not in top 3 sources {sources}")
                if not fact_ok:
                    reasons.append(f"Expected fact '{exp_fact}' not found in answer '{ans}'")
                failures.append({
                    "q_num": q_num,
                    "question": q,
                    "reason": "; ".join(reasons),
                })

            detailed_results.append({
                "q_num": q_num,
                "question": q,
                "expected_doc": exp_doc,
                "expected_fact": exp_fact,
                "sources": sources,
                "answer": ans,
                "doc_retrieved": doc_ok,
                "fact_grounded": fact_ok,
                "status": "PASS" if passed else "FAIL",
            })

    return {
        "total_questions": total_questions,
        "answerable_count": answerable_count,
        "doc_retrieved_count": doc_retrieved_count,
        "fact_grounded_count": fact_grounded_count,
        "unanswerable_count": unanswerable_count,
        "unanswerable_fallback_count": unanswerable_fallback_count,
        "detailed_results": detailed_results,
        "failures": failures,
    }
