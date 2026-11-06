"""Deterministic calculation of policy-governed refund amounts."""
from typing import Dict, Union


def calculate_refund_amount(
    order_amount: Union[float, int],
    delivery_minutes: Union[float, int, None],
    status: str = "completed",
) -> Dict[str, float]:
    """Calculate refund amount according to Instamart refund policy rules.
    
    Rules:
      - Only orders with status 'completed' are eligible.
      - Cancelled orders were never charged -> refund INR 0.
      - Delivered in > 45 minutes -> 100% full refund of order_amount.
      - Delivered in <= 45 minutes -> refund INR 0.
      
    Returns:
      {"refund_inr": <amount>}
    """
    if str(status).lower() != "completed":
        return {"refund_inr": 0.0}

    if delivery_minutes is None:
        return {"refund_inr": 0.0}

    try:
        deliv = float(delivery_minutes)
        amt = float(order_amount)
    except (ValueError, TypeError):
        return {"refund_inr": 0.0}

    if deliv > 45.0:
        return {"refund_inr": round(amt, 2)}
    else:
        return {"refund_inr": 0.0}
