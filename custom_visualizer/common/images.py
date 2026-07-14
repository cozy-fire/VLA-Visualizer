from typing import Any

import numpy as np


def as_image_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    while image.ndim > 3:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"不是图像数组，shape={image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] < 3:
        raise ValueError(f"图像通道数不足，shape={image.shape}")
    image = image[..., :3].astype(np.float32)
    lo, hi = float(image.min()), float(image.max())
    return np.clip((image - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(image)


def collect_image_observations(item: dict[str, Any]) -> tuple[list[str], list[np.ndarray]]:
    images: list[tuple[str, np.ndarray]] = []
    prefix = "observation.images."
    for key in sorted(item):
        value = item[key]
        if key == "observation.images" and isinstance(value, dict):
            for subkey in sorted(value):
                try:
                    images.append((f"{key}.{subkey}", as_image_array(value[subkey])))
                except (TypeError, ValueError, IndexError):
                    continue
            continue
        if not key.startswith(prefix):
            continue
        try:
            images.append((key, as_image_array(value)))
        except (TypeError, ValueError, IndexError):
            continue
    if not images:
        raise ValueError("该数据帧没有 observation.images. 前缀下的可用图像观测")
    keys, arrays = zip(*images)
    return list(keys), list(arrays)