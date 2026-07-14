import json
import os
from pathlib import Path

SETTINGS_PATH = Path(os.environ.get("APPDATA", Path.home())) / "VLM-Visualizer" / "attention_viewer.json"


def load_settings(default_checkpoint: str, default_dataset: str, path: Path = SETTINGS_PATH) -> dict[str, str]:
    settings = {"checkpoint": default_checkpoint, "dataset": default_dataset}
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return settings
    for key in settings:
        value = saved.get(key)
        if isinstance(value, str) and value.strip():
            settings[key] = value.strip()
    return settings


def save_settings(checkpoint: str, dataset: str, path: Path = SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"checkpoint": checkpoint, "dataset": dataset}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)