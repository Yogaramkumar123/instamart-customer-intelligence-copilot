"""Tool definitions for the autonomous refund agent."""
from datetime import datetime, timezone
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Union

from src.part_d.retrieval import retrieve
from src.part_d.refund_calculator import calculate_refund_amount

DEFAULT_DB_PATH = "data/instamart.db"
DEFAULT_LEDGER_PATH = "outputs/refund_ledger.jsonl"


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
    ledger_path: str = DEFAULT_LEDGER_PATH,
) -> Dict[str, float]:
    """Record an approved automated refund in the refund ledger and issue funds."""
    amt = round(float(refund_inr), 2)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "issue_refund",
        "action": "refund_issued",
        "order_id": str(order_id),
        "refund_inr": amt,
    }
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"refund_inr": amt}


def escalate_to_human(
    order_id: str,
    refund_inr: Union[float, int],
    ledger_path: str = DEFAULT_LEDGER_PATH,
) -> Dict[str, float]:
    """Record high-value refund escalation in the refund ledger for human lead approval."""
    amt = round(float(refund_inr), 2)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "escalate_to_human",
        "action": "escalate_to_human",
        "order_id": str(order_id),
        "refund_inr": amt,
    }
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"refund_inr": amt}
