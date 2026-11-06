"""Standalone validation entry point for Part B: Customer Segmentation and Lapse Prediction.

Runs comprehensive checks against:
- RFM metrics
- K-Means evaluations and segment profiles
- PCA explained variance
- Leakage-free lapse cohort and feature matrices
- Model cross-validation and test F1 >= 0.55
- Class-balance oversampling metrics
- Model persistence and reloaded prediction match
- Premium retention list columns and size

Exits with 0 and prints 'ALL PART B CHECKPOINTS PASSED' on success.
Exits with 1 on any assertion failure.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.part_b.validate import validate_part_b


def main():
    try:
        success = validate_part_b()
        if not success:
            sys.exit(1)
    except Exception as exc:
        print(f"\n[VALIDATION FAILED]: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
