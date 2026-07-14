from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DisplayItem:
    entry: dict[str, Any]
    head_idx: int


@dataclass(frozen=True)
class ImagePatchLayout:
    image_key: str
    token_start: int
    token_count: int
    grid_height: int
    grid_width: int


@dataclass(frozen=True)
class TokenLayout:
    image_tokens: int
    language_tokens: int
    state_tokens: int
    prefix_tokens: int
    action_tokens: int
    image_layouts: tuple[ImagePatchLayout, ...]
    overlay_reason: str | None = None

    @property
    def grid_side(self) -> int | None:
        if len(self.image_layouts) != 1:
            return None
        image_layout = self.image_layouts[0]
        if image_layout.grid_height != image_layout.grid_width:
            return None
        return image_layout.grid_height


@dataclass(frozen=True)
class CaptureResult:
    cache: list[dict[str, Any]]
    layout: TokenLayout
    images: list[np.ndarray]
    image_keys: list[str]
    summary: str | None = None