"""Token budget manager for tracking and enforcing usage ceilings."""


class TokenBudgetManager:
    """Tracks token consumption across triage calls and enforces a budget ceiling."""

    def __init__(self, budget: int = 100_000) -> None:
        self.budget: int = budget
        self.total_tokens_used: int = 0

    def is_exhausted(self) -> bool:
        """Check whether the running total of tokens is already >= budget."""
        return self.total_tokens_used >= self.budget

    def record_usage(self, tokens: int) -> None:
        """Record tokens consumed by a successful model interaction."""
        self.total_tokens_used += int(tokens)

    def reset(self) -> None:
        """Reset token consumption counter."""
        self.total_tokens_used = 0
