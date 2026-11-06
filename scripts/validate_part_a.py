"""Validation entry point script for Part A.

Runs the full validation suite against clean datasets and database.
Exits with 0 and prints 'ALL CHECKPOINTS PASSED' on success.
Exits with 1 on any assertion failure.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.part_a.validate import validate_part_a


def main():
    try:
        success = validate_part_a(
            clean_dir=REPO_ROOT / "data" / "clean",
            db_path=REPO_ROOT / "data" / "instamart.db",
            sql_path=REPO_ROOT / "sql" / "queries.sql"
        )
        if not success:
            sys.exit(1)
    except Exception as exc:
        print(f"\n[VALIDATION FAILED]: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
