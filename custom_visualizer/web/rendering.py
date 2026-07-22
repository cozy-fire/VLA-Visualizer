import base64
from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np

from custom_visualizer.common import DisplayItem, TokenLayout, compose_overlay_image


def render_item_png(item: DisplayItem, layout: TokenLayout, source_images: list[np.ndarray], overlay: bool) -> str:
    entry, head = item.entry, item.head_idx
    title = f"S{entry['step']} · L{entry['layer_idx']} · H{head}\n{entry.get('label', entry['type'])}"
    figure = Figure(figsize=(3.6, 3.0), dpi=130)
    axis = figure.subplots()
    if overlay and not layout.overlay_reason:
        axis.imshow(compose_overlay_image(source_images, entry, head, layout))
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    else:
        axis.imshow(np.asarray(entry["probs"])[0, head], cmap="viridis", aspect="auto")
        _draw_partitions(axis, entry, layout)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Key tokens", fontsize=8)
        axis.set_ylabel("Action tokens", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.84)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", facecolor="white")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _draw_partitions(axis: Any, entry: dict[str, Any], layout: TokenLayout) -> None:
    image_end = layout.image_tokens
    language_end = image_end + layout.language_tokens
    axis.axvline(image_end - 0.5, color="white", linewidth=1.2, linestyle="--")
    axis.axvline(language_end - 0.5, color="white", linewidth=1.2, linestyle="--")
    if entry.get("attention_kind", entry["type"]) == "self_attn":
        axis.axvline(layout.prefix_tokens - 0.5, color="red", linewidth=1.6)
