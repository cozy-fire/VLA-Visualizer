import json
import os
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path(os.environ.get("APPDATA", Path.home())) / "VLM-Visualizer" / "attention_viewer.json"
MAX_HISTORY = 20


def load_settings(default_checkpoint: str, default_dataset: str, path: Path = SETTINGS_PATH) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "checkpoint": default_checkpoint,
        "dataset": default_dataset,
        "checkpoint_history": [default_checkpoint] if default_checkpoint else [],
        "dataset_history": [default_dataset] if default_dataset else [],
    }
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return settings

    checkpoint = _clean_string(saved.get("checkpoint")) or default_checkpoint
    dataset = _clean_string(saved.get("dataset")) or default_dataset
    settings["checkpoint"] = checkpoint
    settings["dataset"] = dataset
    settings["checkpoint_history"] = _merge_history(checkpoint, saved.get("checkpoint_history"), default_checkpoint)
    settings["dataset_history"] = _merge_history(dataset, saved.get("dataset_history"), default_dataset)
    return settings


def save_settings(checkpoint: str, dataset: str, path: Path = SETTINGS_PATH) -> None:
    checkpoint = checkpoint.strip()
    dataset = dataset.strip()
    existing = _read_existing(path)
    payload = {
        "checkpoint": checkpoint,
        "dataset": dataset,
        "checkpoint_history": _merge_history(checkpoint, _history_with_current(existing, "checkpoint")),
        "dataset_history": _merge_history(dataset, _history_with_current(existing, "dataset")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_existing(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _history_with_current(existing: dict[str, Any], key: str) -> list[Any]:
    history_key = f"{key}_history"
    history = existing.get(history_key)
    values: list[Any] = [existing.get(key)]
    if isinstance(history, list):
        values.extend(history)
    return values

def _clean_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _merge_history(primary: str | None, saved_history: Any = None, fallback: str | None = None) -> list[str]:
    values: list[str] = []
    for value in [primary, *(saved_history if isinstance(saved_history, list) else []), fallback]:
        cleaned = _clean_string(value)
        if cleaned and cleaned not in values:
            values.append(cleaned)
        if len(values) >= MAX_HISTORY:
            break
    return values