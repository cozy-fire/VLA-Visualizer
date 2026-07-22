import base64
from dataclasses import asdict, is_dataclass
from io import BytesIO
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .types import CaptureResult, DisplayItem, TokenLayout


def ndarray_to_float32_payload(array: Any) -> dict[str, Any]:
    data = np.asarray(array, dtype=np.float32)
    contiguous = np.ascontiguousarray(data)
    return {
        "dtype": "float32",
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def serialize_display_item(item: DisplayItem) -> dict[str, Any]:
    entry = item.entry
    label = str(entry.get("label", entry["type"]))
    return {
        "step": int(entry["step"]),
        "layer": int(entry["layer_idx"]),
        "head": int(item.head_idx),
        "type": str(entry["type"]),
        "label": label,
        "attention_kind": str(entry.get("attention_kind", entry["type"])),
        "mask_type": entry.get("mask_type"),
        "title": f"S{entry['step']} · L{entry['layer_idx']} · H{item.head_idx} · {label}",
        "probs": ndarray_to_float32_payload(np.asarray(entry["probs"])[0, item.head_idx]),
    }


def serialize_layout(layout: TokenLayout) -> dict[str, Any]:
    if is_dataclass(layout):
        return asdict(layout)
    return dict(layout)


def image_to_data_url(image: Any, max_size: int | None = None) -> str:
    array = np.asarray(image, dtype=np.float32)[..., :3]
    array = np.clip(array, 0.0, 1.0)
    pil = Image.fromarray(np.uint8(array * 255))
    if max_size and max(pil.size) > max_size:
        pil.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def serialize_images(images: Iterable[Any], max_size: int | None = None) -> list[dict[str, Any]]:
    result = []
    for image in images:
        array = np.asarray(image)
        result.append({
            "height": int(array.shape[0]),
            "width": int(array.shape[1]),
            "src": image_to_data_url(image, max_size=max_size),
        })
    return result


def serialize_capture_context(result: CaptureResult) -> dict[str, Any]:
    return {
        "layout": serialize_layout(result.layout),
        "source_images": serialize_images(result.images),
        "image_keys": list(result.image_keys),
    }


def summarize_state_value(value: Any, max_values: int = 8) -> dict[str, Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    flat = array.reshape(-1) if array.size else array
    preview = []
    for item in flat[:max_values]:
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float):
            preview.append(round(float(item), 5))
        elif isinstance(item, (int, bool, str)):
            preview.append(item)
        else:
            preview.append(str(item))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "preview": preview,
    }


def summarize_frame_state(item: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for key in sorted(item):
        normalized = key.lower()
        if "state" not in normalized:
            continue
        try:
            summary = summarize_state_value(item[key])
        except Exception:
            continue
        summaries.append({"key": key, **summary})
    return summaries
