"""Document loader for policy text documents."""
import os
from typing import Dict, List

POLICY_FILES = [
    "refund_policy.txt",
    "delivery_policy.txt",
    "payment_policy.txt",
    "privacy_policy.txt",
    "escalation_policy.txt",
]


def load_policy_documents(policies_dir: str = "data/policies") -> Dict[str, str]:
    """Load all five policy documents into a dictionary mapping filename to text content."""
    documents: Dict[str, str] = {}
    for filename in POLICY_FILES:
        filepath = os.path.join(policies_dir, filename)
        assert os.path.exists(filepath), f"Policy document not found: {filepath}"
        with open(filepath, "r", encoding="utf-8") as f:
            documents[filename] = f.read().strip()
    return documents
