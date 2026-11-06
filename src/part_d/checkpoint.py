"""State checkpointing for crash recovery and resumable agent executions."""
import json
import os
from typing import Any, Dict, Optional

DEFAULT_STATE_DIR = "state"


def get_checkpoint_path(run_id: str, state_dir: str = DEFAULT_STATE_DIR) -> str:
    """Get the file path for a run's checkpoint."""
    return os.path.join(state_dir, f"{run_id}.json")


def save_checkpoint(
    run_id: str,
    checkpoint_data: Dict[str, Any],
    state_dir: str = DEFAULT_STATE_DIR,
) -> str:
    """Save an agent execution checkpoint to disk."""
    os.makedirs(state_dir, exist_ok=True)
    filepath = get_checkpoint_path(run_id, state_dir)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
    return filepath


def load_checkpoint(
    run_id: str,
    state_dir: str = DEFAULT_STATE_DIR,
) -> Dict[str, Any]:
    """Load an existing checkpoint from disk."""
    filepath = get_checkpoint_path(run_id, state_dir)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found for run_id '{run_id}' at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
