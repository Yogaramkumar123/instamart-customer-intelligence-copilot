"""PII redaction and verification for support ticket triage."""
import re
from typing import Dict, List, Tuple

# Regex patterns in strict order of application
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_ID_PATTERN = re.compile(r"ORD-\d{6}")

EMAIL_PLACEHOLDER = "[EMAIL]"
PHONE_PLACEHOLDER = "[PHONE]"
ORDER_ID_PLACEHOLDER = "[ORDER_ID]"


def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Redact PII from a single ticket text applying patterns in strict sequence.
    
    Order:
      1. Email -> [EMAIL]
      2. Phone -> [PHONE]
      3. Order ID -> [ORDER_ID]
      
    Returns:
      (redacted_text, counts_dict)
    """
    if not text:
        return text, {"email": 0, "phone": 0, "order_id": 0, "total": 0}

    t_redacted, n_email = EMAIL_PATTERN.subn(EMAIL_PLACEHOLDER, text)
    t_redacted, n_phone = PHONE_PATTERN.subn(PHONE_PLACEHOLDER, t_redacted)
    t_redacted, n_order = ORDER_ID_PATTERN.subn(ORDER_ID_PLACEHOLDER, t_redacted)

    counts = {
        "email": n_email,
        "phone": n_phone,
        "order_id": n_order,
        "total": n_email + n_phone + n_order,
    }
    return t_redacted, counts


def audit_corpus_pii(texts: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """Redact PII across all ticket texts and aggregate counts.
    
    Returns:
      (redacted_texts, total_counts_dict)
    """
    total_counts = {"email": 0, "phone": 0, "order_id": 0, "total": 0}
    redacted_texts: List[str] = []

    for t in texts:
        redacted, counts = redact_pii(t)
        redacted_texts.append(redacted)
        for k in total_counts:
            total_counts[k] += counts[k]

    return redacted_texts, total_counts


def verify_no_pii_remaining(texts: List[str]) -> bool:
    """Verify that re-applying the 3 PII regex patterns yields 0 matches."""
    for t in texts:
        if EMAIL_PATTERN.findall(t):
            return False
        if PHONE_PATTERN.findall(t):
            return False
        if ORDER_ID_PATTERN.findall(t):
            return False
    return True
