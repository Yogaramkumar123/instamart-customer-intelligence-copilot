"""Standalone validation script for Part C: Support Ticket Triage with Guardrails.

Usage:
    python scripts/validate_part_c.py

Prints:
    ALL PART C CHECKPOINTS PASSED
"""
import os
import sys

# Ensure repository root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.part_c.validate import validate_part_c


def main() -> None:
    validate_part_c()


if __name__ == "__main__":
    main()
