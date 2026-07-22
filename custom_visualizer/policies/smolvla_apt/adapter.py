"""SmolVLA-APT Stage 1 attention capture adapter."""

import math
import ssl
from pathlib import Path
from typing import Any, Callable, Sequence

import certifi


_orig_create_default_context = ssl.create_default_context


def _patched_create_default_context(
    purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None,
):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=certifi.where())
    return ctx


ssl.create_default_context = _patched_create_default_context

import numpy as np
import torch

from custom_visualizer.common import (
    CaptureResult,
    ImagePatchLayout,
    SETTINGS_PATH,
    TokenLayout,
    choose_patch_grid,
    collect_image_observations,
    load_lerobot_frame,
)

DEFAULT_CHECKPOINT = (
    r"D:\CodeProject\ModelTrainRepo\Checkpoint\smolvla_apt\stage_1"
    r"\Cozy_pick_pen_give_human\checkpoints\001000\pretrained_model"
)
DEFAULT_DATASET = r"D:\Huggingface Cache\lerobot\CozyFire\pick_pen_give_human"
SETTINGS_PATH_APT = SETTINGS_PATH.with_name("attention_viewer_smolvla_apt.json")


def validate_stage1(config: Any) -> None:
    stage = int(getattr(config, "train_stage", -1))
    if stage != 1:
        raise ValueError(f"SmolVLA-APT Viewer 仅支持 Stage 1 checkpoint，当前 train_stage={stage}")


def reconstruct_action_attention_probs(
    layer: Any,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    action_tokens: int,
    apply_rope_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """Reconstruct the probability part of SDPA for action-token queries."""
    batch_size, sequence_length, hidden_dim = x.shape
    num_heads = int(layer.num_heads)
    head_dim = hidden_dim // num_heads
    if action_tokens <= 0 or action_tokens > sequence_length:
        raise ValueError(
            f"action_tokens={action_tokens} 与序列长度 {sequence_length} 不兼容"
        )

    normalized = layer.norm1(x)
    q = layer.q_proj(normalized).view(batch_size, sequence_length, num_heads, head_dim)
    k = layer.k_proj(normalized).view(batch_size, sequence_length, num_heads, head_dim)
    if position_ids is not None:
        if apply_rope_fn is None:
            raise ValueError("position_ids 非空时必须提供 apply_rope_fn")
        q = apply_rope_fn(q, position_ids)
        k = apply_rope_fn(k, position_ids)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    mask = attention_mask.to(dtype=torch.bool)
    if mask.ndim != 3 or mask.shape[-2:] != (sequence_length, sequence_length):
        raise ValueError(
            f"attention_mask shape={tuple(mask.shape)}，期望 (B, {sequence_length}, {sequence_length})"
        )
    fully_masked = ~mask.any(dim=-1)
    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=x.device)[None, :, :]
    mask = mask | (fully_masked.unsqueeze(-1) & diagonal)

    q_action = q[:, :, -action_tokens:, :]
    action_mask = mask[:, -action_tokens:, :].unsqueeze(1)
    logits = torch.matmul(q_action.float(), k.float().transpose(-1, -2))
    logits = logits / math.sqrt(head_dim)
    logits = logits.masked_fill(~action_mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def make_apt_cache_entry(step: int, layer_idx: int, probs: np.ndarray) -> dict[str, Any]:
    is_likelihood = layer_idx % 2 == 0
    return {
        "phase": 2,
        "step": int(step),
        "layer_idx": int(layer_idx),
        "type": "vla_likelihood" if is_likelihood else "va_prior",
        "label": "VLA似然" if is_likelihood else "VA先验",
        "attention_kind": "self_attn",
        "mask_type": "full" if is_likelihood else "dilated",
        "probs": probs,
    }


def build_apt_token_layout(
    cache: Sequence[dict[str, Any]],
    patch_counts: Sequence[int],
    image_keys: Sequence[str],
    images: Sequence[np.ndarray],
    language_tokens: int,
    add_image_special_tokens: bool,
) -> TokenLayout:
    if not cache:
        raise ValueError("SmolVLA-APT attention cache 为空")
    sample = cache[0]["probs"]
    action_tokens = int(sample.shape[-2])
    key_tokens = int(sample.shape[-1])
    prefix_tokens = key_tokens - action_tokens
    state_tokens = 1

    reason = None
    image_layouts: list[ImagePatchLayout] = []
    if len(patch_counts) != len(images) or len(images) != len(image_keys):
        reason = (
            f"图像、image key 与 patch token 分块数量不一致: "
            f"images={len(images)}, keys={len(image_keys)}, blocks={len(patch_counts)}"
        )
        image_tokens = prefix_tokens - int(language_tokens) - state_tokens
    else:
        cursor = 0
        for key, image, patch_count in zip(image_keys, images, patch_counts, strict=True):
            patch_count = int(patch_count)
            if add_image_special_tokens:
                cursor += 1
            token_start = cursor
            height, width = image.shape[:2]
            grid_height, grid_width = choose_patch_grid(patch_count, height, width)
            image_layouts.append(ImagePatchLayout(
                image_key=key,
                token_start=token_start,
                token_count=patch_count,
                grid_height=grid_height,
                grid_width=grid_width,
            ))
            cursor += patch_count
            if add_image_special_tokens:
                cursor += 1
        image_tokens = cursor

    language_region = prefix_tokens - image_tokens - state_tokens
    if image_tokens <= 0 or language_region < int(language_tokens):
        reason = reason or (
            f"无法推导 APT token 布局: prefix={prefix_tokens}, image={image_tokens}, "
            f"language={language_tokens}, state={state_tokens}"
        )
        image_layouts = []
        image_tokens = max(0, prefix_tokens - int(language_tokens) - state_tokens)
        language_region = max(0, prefix_tokens - image_tokens - state_tokens)

    return TokenLayout(
        image_tokens=int(image_tokens),
        language_tokens=int(language_region),
        state_tokens=state_tokens,
        prefix_tokens=prefix_tokens,
        action_tokens=action_tokens,
        image_layouts=tuple(image_layouts),
        overlay_reason=reason,
    )


def _ordered_image_keys(config: Any, available_keys: Sequence[str]) -> list[str]:
    configured = list(getattr(config, "image_features", {}) or [])
    keys = configured or list(available_keys)
    camera_order = list(getattr(config, "camera_order", None) or [])
    if camera_order:
        order = {key: idx for idx, key in enumerate(camera_order)}
        keys = sorted(enumerate(keys), key=lambda pair: (order.get(pair[1], len(order)), pair[0]))
        keys = [key for _, key in keys]
    return keys


def _placeholder_shape(config: Any, images: Sequence[np.ndarray]) -> tuple[int, int]:
    resize = getattr(config, "resize_imgs_with_padding", None)
    if isinstance(resize, (tuple, list)) and len(resize) >= 2:
        return int(resize[0]), int(resize[1])
    if images:
        return int(images[0].shape[0]), int(images[0].shape[1])
    return 224, 224


def collect_apt_source_images(item: dict[str, Any], config: Any) -> tuple[list[str], list[np.ndarray]]:
    available_keys, available_images = collect_image_observations(item)
    by_key = dict(zip(available_keys, available_images, strict=True))
    ordered_keys = _ordered_image_keys(config, available_keys)
    height, width = _placeholder_shape(config, available_images)
    images = [
        by_key[key] if key in by_key else np.zeros((height, width, 3), dtype=np.float32)
        for key in ordered_keys
    ]
    return ordered_keys, images


class SmolVLAAptAdapter:
    name = "SmolVLA-APT"
    default_checkpoint = DEFAULT_CHECKPOINT
    default_dataset = DEFAULT_DATASET
    settings_path: Path = SETTINGS_PATH_APT
    restart_required_message = "切换 SmolVLA-APT checkpoint 后需要重启 Viewer 服务"

    def can_reuse_context(self, context: dict[str, Any] | None, checkpoint: str) -> bool:
        return bool(context and context.get("checkpoint") == checkpoint)

    def load_context(self, checkpoint: str) -> dict[str, Any]:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla_apt.modeling_smolvla_apt import SmolVLAAptPolicy
        from lerobot.policies.smolvla_apt.smolvlm_with_expert import apply_rope

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = SmolVLAAptPolicy.from_pretrained(checkpoint)
        validate_stage1(policy.config)
        policy.to(device)
        policy.eval()
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        return {
            "checkpoint": checkpoint,
            "policy": policy,
            "preprocessor": preprocessor,
            "apply_rope": apply_rope,
        }

    def capture(self, context: dict[str, Any], dataset_repo: str, frame_idx: int) -> CaptureResult:
        policy = context["policy"]
        validate_stage1(policy.config)
        item = load_lerobot_frame(policy.config, dataset_repo, frame_idx)
        image_keys, images = collect_apt_source_images(item, policy.config)
        batch = context["preprocessor"](item)
        language_tokens = int(batch["observation.language.tokens"].shape[-1])

        model = policy.model
        layers = list(model.hybrid_attn_layers.layers)
        if not layers:
            raise RuntimeError("Stage 1 HybridAttentionLayers 中没有 active layer")
        action_tokens = int(policy.config.chunk_size)
        expected_steps = int(policy.config.num_steps)
        cache: list[dict[str, Any]] = []
        step_counts = [0] * len(layers)
        handles = []
        patch_counts: list[int] = []

        def make_hook(layer_idx: int):
            def hook(module, args, kwargs):
                x = kwargs.get("x", args[0] if args else None)
                attention_mask = kwargs.get("attention_mask")
                position_ids = kwargs.get("position_ids")
                if x is None or attention_mask is None:
                    raise RuntimeError("无法从 AttentionBlock hook 获取 x 或 attention_mask")
                probs = reconstruct_action_attention_probs(
                    module, x, attention_mask, position_ids, action_tokens, context["apply_rope"]
                )
                step = step_counts[layer_idx]
                step_counts[layer_idx] += 1
                cache.append(make_apt_cache_entry(
                    step, layer_idx, probs.detach().cpu().float().numpy()
                ))
            return hook

        for layer_idx, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))

        vlm = model.vlm_with_expert
        original_embed_image = vlm.embed_image

        def recording_embed_image(image):
            embeddings = original_embed_image(image)
            patch_counts.append(int(embeddings.shape[1]))
            return embeddings

        vlm.embed_image = recording_embed_image
        try:
            with torch.no_grad():
                policy.predict_action_chunk(batch)
        finally:
            vlm.embed_image = original_embed_image
            for handle in handles:
                handle.remove()

        if not cache:
            raise RuntimeError("未捕获到 SmolVLA-APT Stage 1 Hybrid Attention")
        if any(count != expected_steps for count in step_counts):
            raise RuntimeError(
                f"Hybrid Attention 调用次数异常: expected={expected_steps}, actual={step_counts}"
            )
        cache.sort(key=lambda entry: (entry["step"], entry["layer_idx"]))
        layout = build_apt_token_layout(
            cache=cache,
            patch_counts=patch_counts,
            image_keys=image_keys,
            images=images,
            language_tokens=language_tokens,
            add_image_special_tokens=bool(policy.config.add_image_special_tokens),
        )
        return CaptureResult(
            cache=cache,
            layout=layout,
            images=images,
            image_keys=image_keys,
            summary=(
                f"SmolVLA-APT Stage 1: steps={expected_steps}, layers={len(layers)}, "
                f"heads={cache[0]['probs'].shape[1]}, images={len(image_keys)}"
            ),
        )
