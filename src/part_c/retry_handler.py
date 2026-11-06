"""Retry handler with exponential backoff and jitter for mock LLM calls."""
import random
import time
from typing import Any, Callable, Dict, List, Optional
import mock_llm

BASE_DELAY = 0.05
MAX_RETRIES = 3


def call_with_retry(
    messages: List[Dict[str, Any]],
    ticket_id: str = "UNKNOWN",
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    log_retry: bool = True,
    retry_records: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Execute mock_llm.chat with exponential backoff and jitter on TransientLLMError.
    
    Retry index n is 0, 1, 2 for retries 1, 2, 3.
    Wait time: base_delay * (2 ** n) * random.uniform(0.9, 1.15)
    
    Args:
        messages: The chat messages payload.
        ticket_id: Identifier for logging and traceability.
        max_retries: Maximum number of retries before giving up.
        base_delay: Base delay in seconds (0.05s).
        log_retry: Whether to print retry log line.
        retry_records: Optional list to append retry audit details.
        
    Returns:
        Chat response from mock_llm.chat.
    """
    for n in range(max_retries + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError as err:
            if n >= max_retries:
                # Exceeded maximum retries
                raise err
            
            # Compute wait with exponential backoff + jitter
            wait = base_delay * (2 ** n) * random.uniform(0.9, 1.15)
            log_line = f"[RETRY] Ticket={ticket_id} | retry={n} | wait={wait:.4f}s"
            if log_retry:
                print(log_line)
            
            if retry_records is not None:
                retry_records.append({
                    "ticket_id": ticket_id,
                    "retry_n": n,
                    "wait": wait,
                    "error": str(err),
                })
            
            time.sleep(wait)
