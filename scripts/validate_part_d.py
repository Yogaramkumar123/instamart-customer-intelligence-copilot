"""Standalone validation script for Part D: Policy RAG Assistant and Refund Agent.

Usage:
    python scripts/validate_part_d.py

Prints:
    ==================================================
    ALL PART D CHECKPOINTS PASSED
    ==================================================
"""
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.part_d.validate import validate_part_d


def main() -> None:
    validate_part_d()


if __name__ == "__main__":
    main()
