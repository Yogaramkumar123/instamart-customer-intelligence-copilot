"""Classifier orchestrating guardrails, redaction, model call, and validation."""
from typing import Any, Dict, List, Optional, Tuple
from src.part_c.guardrails import check_input_guardrails
from src.part_c.pii_redaction import redact_pii, verify_no_pii_remaining
from src.part_c.retry_handler import call_with_retry
from src.part_c.output_validation import validate_model_output
from src.part_c.token_budget import TokenBudgetManager


def classify_ticket(
    ticket_id: str,
    raw_text: str,
    system_prompt: str,
    budget_manager: TokenBudgetManager,
    log_retry: bool = False,
    retry_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[str], int]:
    """Process a single ticket through guardrails, redaction, retryable LLM, and validation.
    
    Returns:
        (status, category, tokens_used)
    """
    # 1. Input Guardrails
    guardrail_status = check_input_guardrails(raw_text)
    if guardrail_status is not None:
        return guardrail_status, None, 0

    # 2. PII Redaction
    redacted_text, _ = redact_pii(raw_text)
    assert verify_no_pii_remaining([redacted_text]), f"PII leakage detected in ticket {ticket_id}!"

    # 3. Token Budget Check
    if budget_manager.is_exhausted():
        return "budget_exhausted", None, 0

    # 4. Message formulation - strictly redacted text only as user content
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text},
    ]

    # 5. Model call with retry handler
    response = call_with_retry(
        messages=messages,
        ticket_id=ticket_id,
        log_retry=log_retry,
        retry_records=retry_records,
    )
    tokens_used = response.usage.total_tokens
    budget_manager.record_usage(tokens_used)

    # 6. Output validation
    content = response.choices[0].message.content
    status, category = validate_model_output(content)

    return status, category, tokens_used
