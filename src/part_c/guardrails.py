"""Input guardrails for support ticket triage."""
from typing import Optional
import mock_llm

MAX_INPUT_TOKENS: int = 400


def check_input_guardrails(text: str) -> Optional[str]:
    """Check input ticket text against pre-model guardrails.
    
    Returns:
        'skipped_empty' if text is empty or whitespace only.
        'flagged_long' if est_tokens(text) > MAX_INPUT_TOKENS (400).
        None if the ticket passes all guardrails.
    """
    if text is None or not str(text).strip():
        return "skipped_empty"
    
    tokens = mock_llm.est_tokens(str(text))
    if tokens > MAX_INPUT_TOKENS:
        return "flagged_long"
    
    return None
