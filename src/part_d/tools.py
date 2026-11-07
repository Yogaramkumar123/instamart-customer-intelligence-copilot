"""Tool definitions for the autonomous refund agent with financial safety and idempotency."""
from datetime import datetime, timezone
import json
import math
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Union
import uuid

from src.part_d.retrieval import retrieve
from src.part_d.refund_calculator import calculate_refund_amount

DEFAULT_DB_PATH = "data/instamart.db"
DEFAULT_LEDGER_PATH = "outputs/refund_ledger.jsonl"

_LEDGER_LOCK = threading.Lock()


def get_order(order_id: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Retrieve order details from the relational SQLite database.

    Returns:
        {"order_id": str, "status": str, "order_amount": float, "delivery_minutes": int}
        or {"error": "order not found"}
    """
    if not os.path.exists(db_path):
        return {"error": "database not found"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"error": "order not found"}

    oid, status, amount, deliv_mins = row
    parsed_deliv = int(deliv_mins) if (deliv_mins is not None and deliv_mins != "") else None

    return {
        "order_id": str(oid),
        "status": str(status),
        "order_amount": float(amount),
        "delivery_minutes": parsed_deliv,
    }


def search_policy(query: str, k: int = 3) -> Dict[str, Any]:
    """Retrieve grounded policy chunks relevant to the query."""
    results = retrieve(query, k=k)
    return {"results": results}


def calculate_refund(
    order_amount: Union[float, int],
    delivery_minutes: Union[float, int, None],
) -> Dict[str, float]:
    """Calculate refund amount based on order delivery latency and completion status."""
    return calculate_refund_amount(order_amount=order_amount, delivery_minutes=delivery_minutes)


def issue_refund(
    order_id: str,
    refund_inr: Union[float, int],
    idempotency_key: Optional[str] = None,
    ledger_path: str = DEFAULT_LEDGER_PATH,
) -> Dict[str, Any]:
    """Record an approved automated refund in the refund ledger with financial safety and idempotency.

    Validation & Safety Rules:
    - Order ID must be non-empty string.
    - Amount must be numeric, finite (not NaN, not Inf), non-negative (>= 0).
    - Amount must not exceed the auto-approval cap of INR 1000.00.
    - If refund amount exceeds 1000.00, it is rejected and NOT appended to the ledger.
    - Duplicate idempotency keys are detected; repeated requests for the same transaction
      will NOT issue a second payout or append a duplicate record to the ledger.
    """
    if order_id is None or not str(order_id).strip():
        return {"error": "Invalid order_id: must be a non-empty string.", "status": "rejected"}

    if refund_inr is None:
        return {"error": "Refund amount cannot be null.", "status": "rejected"}

    try:
        amt_float = float(refund_inr)
    except (TypeError, ValueError):
        return {"error": f"Invalid refund amount '{refund_inr}': must be numeric.", "status": "rejected"}

    if not math.isfinite(amt_float):
        return {"error": f"Invalid refund amount '{refund_inr}': must be a finite number.", "status": "rejected"}

    if amt_float < 0:
        return {"error": f"Invalid refund amount {amt_float}: amount cannot be negative.", "status": "rejected"}

    amt = round(amt_float, 2)

    if amt > 1000.00:
        return {
            "error": f"Refund amount INR {amt:.2f} exceeds maximum auto-approval limit of INR 1000.00.",
            "status": "rejected",
        }

    # Derive or sanitize transaction-level idempotency key
    if idempotency_key is None or not str(idempotency_key).strip():
        key = f"tx_{order_id}_{uuid.uuid4().hex[:12]}"
    else:
        key = str(idempotency_key).strip()

    os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)

    with _LEDGER_LOCK:
        # Check existing records in the ledger for duplicate idempotency key
        if os.path.exists(ledger_path):
            with open(ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("tool") == "issue_refund" and entry.get("idempotency_key") == key:
                        return {
                            "error": f"Duplicate transaction: refund already processed for idempotency_key '{key}'.",
                            "status": "duplicate",
                            "refund_inr": entry.get("refund_inr", amt),
                            "idempotency_key": key,
                            "original_timestamp": entry.get("timestamp"),
                        }

        # Not a duplicate: record transaction to ledger
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": "issue_refund",
            "action": "refund_issued",
            "order_id": str(order_id),
            "refund_inr": amt,
            "idempotency_key": key,
        }
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    return {
        "refund_inr": amt,
        "status": "issued",
        "idempotency_key": key,
    }


def escalate_to_human(
    order_id: str,
    refund_inr: Union[float, int],
    idempotency_key: Optional[str] = None,
    ledger_path: str = DEFAULT_LEDGER_PATH,
) -> Dict[str, Any]:
    """Record high-value refund escalation in the refund ledger for human lead approval.

    Validation & Safety Rules:
    - Order ID must be non-empty string.
    - Amount must be numeric, finite, non-negative.
    - Supports amounts exceeding INR 1000.00 (which mandate supervisor escalation).
    """
    if order_id is None or not str(order_id).strip():
        return {"error": "Invalid order_id: must be a non-empty string.", "status": "rejected"}

    if refund_inr is None:
        return {"error": "Escalation refund amount cannot be null.", "status": "rejected"}

    try:
        amt_float = float(refund_inr)
    except (TypeError, ValueError):
        return {"error": f"Invalid escalation refund amount '{refund_inr}': must be numeric.", "status": "rejected"}

    if not math.isfinite(amt_float):
        return {"error": f"Invalid escalation refund amount '{refund_inr}': must be a finite number.", "status": "rejected"}

    if amt_float < 0:
        return {"error": f"Invalid escalation refund amount {amt_float}: amount cannot be negative.", "status": "rejected"}

    amt = round(amt_float, 2)

    if idempotency_key is None or not str(idempotency_key).strip():
        key = f"esc_{order_id}_{uuid.uuid4().hex[:12]}"
    else:
        key = str(idempotency_key).strip()

    os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)

    with _LEDGER_LOCK:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": "escalate_to_human",
            "action": "escalate_to_human",
            "order_id": str(order_id),
            "refund_inr": amt,
            "idempotency_key": key,
        }
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    return {
        "refund_inr": amt,
        "status": "escalated",
        "idempotency_key": key,
    }
