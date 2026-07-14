import math
from typing import Any, Iterable, Sequence

import numpy as np

from .types import DisplayItem, ImagePatchLayout, TokenLayout

ALL = "全部"
STATIC_COLUMNS = 4
STATIC_VISIBLE_ROWS = 3


def phase2_entries(cache: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in cache if entry["phase"] == 2]


def common_head_indices(cache: Iterable[dict[str, Any]]) -> list[int]:
    entries = phase2_entries(cache)
    return [] if not entries else list(range(min(int(e["probs"].shape[1]) for e in entries)))


def filter_display_items(
    cache: Iterable[dict[str, Any]], step: int | None = None,
    layer: int | None = None, head: int | None = None,
) -> list[DisplayItem]:
    items: list[DisplayItem] = []
    for entry in phase2_entries(cache):
        if step is not None and entry["step"] != step:
            continue
        if layer is not None and entry["layer_idx"] != layer:
            continue
        heads = [head] if head is not None else range(entry["probs"].shape[1])
        for head_idx in heads:
            if 0 <= head_idx < entry["probs"].shape[1]:
                items.append(DisplayItem(entry, int(head_idx)))
    items.sort(key=lambda x: (x.entry["step"], x.entry["layer_idx"], x.head_idx))
    return items


def static_window(
    items: Sequence[DisplayItem], start_row: int,
    columns: int = STATIC_COLUMNS, visible_rows: int = STATIC_VISIBLE_ROWS,
) -> tuple[int, int, list[DisplayItem]]:
    total_rows = max(1, math.ceil(len(items) / columns))
    max_start_row = max(0, total_rows - visible_rows)
    start_row = min(max(0, int(start_row)), max_start_row)
    start = start_row * columns
    stop = start + columns * visible_rows
    return start_row, total_rows, list(items[start:stop])


def build_dynamic_frames(
    cache: Iterable[dict[str, Any]], mode: str, step: int | None = None,
    layer: int | None = None, head: int | None = None,
) -> list[tuple[int, list[DisplayItem]]]:
    axis_by_mode = {"steps": "step", "layers": "layer", "heads": "head"}
    if mode not in axis_by_mode:
        raise ValueError(f"未知动态模式: {mode}")
    values = {"step": step, "layer": layer, "head": head}
    axis = axis_by_mode[mode]
    if values[axis] is not None:
        raise ValueError(f"{mode} 模式下 {axis} 必须保持“全部”")
    fixed = [name for name in values if name != axis and values[name] is not None]
    if len(fixed) != 1:
        raise ValueError("动态模式下，另外两个参数必须且只能指定一个")

    entries = phase2_entries(cache)
    if axis == "step":
        axis_values = sorted({int(e["step"]) for e in entries})
    elif axis == "layer":
        axis_values = sorted({int(e["layer_idx"]) for e in entries})
    else:
        axis_values = common_head_indices(entries)
    frames = []
    for axis_value in axis_values:
        args = dict(step=step, layer=layer, head=head)
        args[axis] = axis_value
        items = filter_display_items(cache, **args)
        if items:
            frames.append((axis_value, items))
    return frames


def choose_patch_grid(token_count: int, image_height: int = 1, image_width: int = 1) -> tuple[int, int]:
    token_count = int(token_count)
    if token_count <= 0:
        raise ValueError("token_count 必须大于 0")
    image_height = max(1, int(image_height))
    image_width = max(1, int(image_width))
    target_ratio = image_width / image_height
    best_grid = (1, token_count)
    best_error = float("inf")
    for height in range(1, math.isqrt(token_count) + 1):
        if token_count % height:
            continue
        width = token_count // height
        for grid_height, grid_width in ((height, width), (width, height)):
            ratio = grid_width / grid_height
            error = abs(math.log(ratio / target_ratio))
            if error < best_error:
                best_grid = (grid_height, grid_width)
                best_error = error
    return best_grid


def infer_token_layout(
    cache: Iterable[dict[str, Any]], language_tokens: int,
    image_observation_count: int = 1, state_tokens: int = 1,
    image_shapes: Sequence[tuple[int, int]] | None = None,
    image_keys: Sequence[str] | None = None,
) -> TokenLayout:
    entries = phase2_entries(cache)
    if not entries:
        raise ValueError("Phase 2 cache 为空")
    cross = next((e for e in entries if e["type"] == "expert_cross_attn"), None)
    sample = cross or entries[0]
    action_tokens = int(sample["probs"].shape[-2])
    key_tokens = int(sample["probs"].shape[-1])
    prefix_tokens = key_tokens if cross else key_tokens - action_tokens
    image_tokens = prefix_tokens - int(language_tokens) - int(state_tokens)
    if image_tokens <= 0:
        raise ValueError(
            f"无法推导 image token: prefix={prefix_tokens}, language={language_tokens}, state={state_tokens}"
        )

    count = max(1, int(image_observation_count))
    shapes = list(image_shapes or [(1, 1)] * count)
    keys = list(image_keys or [f"image[{idx}]" for idx in range(count)])
    reason = None
    image_layouts: tuple[ImagePatchLayout, ...] = ()
    if image_tokens % count:
        reason = f"image token 数 {image_tokens} 不能平均分给 {count} 个图像观测"
    else:
        tokens_per_image = image_tokens // count
        layouts = []
        for idx in range(count):
            height, width = shapes[idx] if idx < len(shapes) else (1, 1)
            grid_height, grid_width = choose_patch_grid(tokens_per_image, height, width)
            layouts.append(ImagePatchLayout(
                image_key=keys[idx] if idx < len(keys) else f"image[{idx}]",
                token_start=idx * tokens_per_image,
                token_count=tokens_per_image,
                grid_height=grid_height,
                grid_width=grid_width,
            ))
        image_layouts = tuple(layouts)
    return TokenLayout(image_tokens, int(language_tokens), int(state_tokens),
                       prefix_tokens, action_tokens, image_layouts, reason)


def extract_patch_scores(entry: dict[str, Any], head_idx: int, layout: TokenLayout) -> np.ndarray:
    probs = np.asarray(entry["probs"])
    if not 0 <= head_idx < probs.shape[1]:
        raise IndexError(f"head_idx={head_idx} 越界")
    return probs[0, head_idx, :, :layout.image_tokens].mean(axis=0)


def extract_view_patch_scores(
    entry: dict[str, Any], head_idx: int,
    layout: TokenLayout, image_layout: ImagePatchLayout,
) -> np.ndarray:
    scores = extract_patch_scores(entry, head_idx, layout)
    start = image_layout.token_start
    stop = start + image_layout.token_count
    return scores[start:stop]


def patch_scores_to_image_grid(scores: np.ndarray, image_layout: ImagePatchLayout) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size != image_layout.token_count:
        raise ValueError(
            f"patch score 数 {scores.size} 与 {image_layout.image_key} token 数 {image_layout.token_count} 不一致"
        )
    lo, hi = float(scores.min()), float(scores.max())
    normalized = np.zeros_like(scores) if hi <= lo else (scores - lo) / (hi - lo)
    return normalized.reshape(image_layout.grid_height, image_layout.grid_width)


def patch_scores_to_grid(scores: np.ndarray, layout: TokenLayout) -> np.ndarray:
    if not layout.image_layouts:
        raise ValueError(layout.overlay_reason or "image token 不能组成二维网格")
    return patch_scores_to_image_grid(scores[:layout.image_layouts[0].token_count], layout.image_layouts[0])


def overlay_single_image(
    source_image: np.ndarray, patch_scores: np.ndarray,
    image_layout: ImagePatchLayout, alpha: float = 0.5,
) -> np.ndarray:
    from PIL import Image
    from matplotlib import colormaps

    source = np.asarray(source_image, dtype=np.float32)[..., :3]
    grid = patch_scores_to_image_grid(patch_scores, image_layout)
    height, width = source.shape[:2]
    resampling = getattr(Image, "Resampling", Image)
    heatmap = np.asarray(
        Image.fromarray(np.uint8(grid * 255)).resize((width, height), resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    colors = colormaps.get_cmap("jet")(heatmap)[..., :3].astype(np.float32)
    return np.clip(source * (1.0 - alpha) + colors * alpha, 0.0, 1.0)


def compose_overlay_image(
    source_images: Sequence[np.ndarray], entry: dict[str, Any],
    head_idx: int, layout: TokenLayout, alpha: float = 0.5,
) -> np.ndarray:
    if layout.overlay_reason:
        raise ValueError(layout.overlay_reason)
    if len(source_images) != len(layout.image_layouts):
        raise ValueError(
            f"图像数量 {len(source_images)} 与布局数量 {len(layout.image_layouts)} 不一致"
        )
    overlays = [
        overlay_single_image(
            source_image, extract_view_patch_scores(entry, head_idx, layout, image_layout),
            image_layout, alpha=alpha,
        )
        for source_image, image_layout in zip(source_images, layout.image_layouts)
    ]
    if len(overlays) == 1:
        return overlays[0]
    max_height = max(image.shape[0] for image in overlays)
    padded = []
    for image in overlays:
        if image.shape[0] == max_height:
            padded.append(image)
            continue
        canvas = np.ones((max_height, image.shape[1], 3), dtype=np.float32)
        canvas[:image.shape[0], :image.shape[1]] = image
        padded.append(canvas)
    return np.concatenate(padded, axis=1)