"""Local snapshot storage using JSON files in .prompt-drift/ directory."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from .models import Snapshot


DEFAULT_STORE_DIR = ".prompt-drift"


def _store_path(base_dir: Optional[str] = None) -> Path:
    """Resolve the snapshot store directory."""
    if base_dir:
        p = Path(base_dir)
    else:
        p = Path.cwd() / DEFAULT_STORE_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _snapshot_file(name: str, base_dir: Optional[str] = None) -> Path:
    """Return the file path for a named snapshot."""
    if not name or not name.strip():
        raise ValueError("Snapshot name must not be empty")
    safe_name = name.replace("/", "_").replace("\\", "_")
    return _store_path(base_dir) / f"{safe_name}.json"


def save_snapshot(snapshot: Snapshot, base_dir: Optional[str] = None) -> Path:
    """Save a snapshot to the store.

    Args:
        snapshot: The Snapshot object to persist.
        base_dir: Override the store directory path.

    Returns:
        Path to the saved JSON file.
    """
    path = _snapshot_file(snapshot.name, base_dir)
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_snapshot(name: str, base_dir: Optional[str] = None) -> Snapshot:
    """Load a snapshot by name.

    Raises:
        FileNotFoundError: If the snapshot does not exist.
    """
    path = _snapshot_file(name, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot '{name}' not found at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Snapshot '{name}' contains invalid JSON: {exc}") from exc
    return Snapshot(**data)


def list_snapshots(base_dir: Optional[str] = None) -> List[str]:
    """List all snapshot names in the store."""
    store = _store_path(base_dir)
    names: List[str] = []
    for f in sorted(store.glob("*.json")):
        names.append(f.stem)
    return names


def delete_snapshot(name: str, base_dir: Optional[str] = None) -> bool:
    """Delete a snapshot by name. Returns True if deleted, False if not found."""
    path = _snapshot_file(name, base_dir)
    if path.exists():
        path.unlink()
        return True
    return False


def snapshot_exists(name: str, base_dir: Optional[str] = None) -> bool:
    """Check whether a snapshot with the given name exists."""
    return _snapshot_file(name, base_dir).exists()
