"""Customer segment lookup and routing logic for support ticket triage."""
from typing import Any, Dict, Optional, Tuple
import pandas as pd

UNCLASSIFIED_STATUSES = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted",
}


def load_customer_segments(segments_path: str = "outputs/customer_segments.csv") -> Dict[str, Dict[str, Any]]:
    """Load customer segments lookup table keyed by customer_id."""
    df = pd.read_csv(segments_path)
    lookup: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        lookup[str(row["customer_id"])] = {
            "segment_name": str(row["segment_name"]),
            "is_premium": bool(row["is_premium"]),
        }
    return lookup


def route_ticket(
    status: str,
    category: Optional[str],
    customer_id: str,
    segments_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[str, str, str, str]:
    """Determine routing and priority for a ticket based on triage status and customer segment.
    
    Routing rules:
      1. If status in ('skipped_empty', 'flagged_long', 'flagged_invalid_output', 'budget_exhausted')
         -> route: 'human_review', priority: 'NORMAL'
      2. Else if category == 'fraud'
         -> route: 'security_team', priority: 'HIGH'
      3. Else if customer's is_premium is True
         -> route: 'premium_desk', priority: 'HIGH'
      4. Else
         -> route: 'support_queue', priority: 'NORMAL'
         
    Returns:
      (category_str, segment_name_str, priority, route)
    """
    customer_info = segments_lookup.get(customer_id)
    if customer_info is not None:
        segment_name = customer_info["segment_name"]
        is_premium = customer_info["is_premium"]
    else:
        segment_name = ""
        is_premium = False

    cat_str = category if category else ""

    if status in UNCLASSIFIED_STATUSES:
        return "", segment_name, "NORMAL", "human_review"

    if cat_str.lower() == "fraud":
        return cat_str, segment_name, "HIGH", "security_team"

    if is_premium:
        return cat_str, segment_name, "HIGH", "premium_desk"

    return cat_str, segment_name, "NORMAL", "support_queue"
