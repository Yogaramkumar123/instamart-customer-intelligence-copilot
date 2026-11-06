"""Output validation and keyword fallback for support ticket triage."""
import json
import re
from typing import Optional, Set, Tuple

ALLOWED_CATEGORIES: Set[str] = {"billing", "delivery", "technical", "fraud"}


def validate_model_output(content: str) -> Tuple[str, Optional[str]]:
    """Validate model response text according to task 5 rules.
    
    1. Parse reply as JSON. If category in ALLOWED_CATEGORIES:
       status = 'classified_json', category = parsed['category'].
       If JSON is valid but category is outside the allowed list:
       status = 'flagged_invalid_output', category = None.
       
    2. If not valid JSON:
       Extract lowercase words with re.findall(r"[a-z]+", content.lower()).
       If exactly one allowed category word is present:
       status = 'classified_keyword', category = word.
       Otherwise:
       status = 'flagged_invalid_output', category = None.
       
    Returns:
        (status, category)
    """
    if not content:
        return "flagged_invalid_output", None

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            cat = str(data.get("category", "")).lower()
            if cat in ALLOWED_CATEGORIES:
                return "classified_json", cat
            else:
                return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Fallback to word extraction for non-JSON answers
    words = re.findall(r"[a-z]+", content.lower())
    matched_cats = [w for w in words if w in ALLOWED_CATEGORIES]

    if len(matched_cats) == 1:
        return "classified_keyword", matched_cats[0]

    return "flagged_invalid_output", None
