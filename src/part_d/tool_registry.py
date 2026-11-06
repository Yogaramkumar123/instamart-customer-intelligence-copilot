"""Tool specifications and dispatch map for the autonomous refund agent."""
from typing import Any, Callable, Dict, List
from src.part_d.tools import (
    get_order,
    search_policy,
    calculate_refund,
    issue_refund,
    escalate_to_human,
)

TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "get_order",
        "description": "Look up an order record by its order_id from the instamart analytical database.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Unique identifier of the order (e.g., ORD-100048).",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Search operational company policy documents using TF-IDF cosine similarity retrieval.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language policy query string.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate eligible refund amount in INR based on order amount and delivery duration.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {
                    "type": "number",
                    "description": "Total order spend in INR.",
                },
                "delivery_minutes": {
                    "type": "integer",
                    "description": "Duration taken for delivery in whole minutes.",
                },
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue an approved refund up to INR 1000 and record it in the ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order identifier to receive the refund.",
                },
                "refund_inr": {
                    "type": "number",
                    "description": "Refund amount in INR to be issued.",
                },
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate a high-value refund request exceeding INR 1000 to a human support lead.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order identifier for the high-value refund escalation.",
                },
                "refund_inr": {
                    "type": "number",
                    "description": "Refund amount in INR requiring human approval.",
                },
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

TOOL_MAP: Dict[str, Callable[..., Any]] = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}
