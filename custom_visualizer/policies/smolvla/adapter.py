"""SmolVLA-specific attention capture adapter."""

import ssl
import types
from typing import Any, Sequence

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
    collect_image_observations,
    infer_token_layout,
    load_lerobot_frame,
)

CACHE_KEY = "SmolVLMWithExpertModel.eager_attention_forward"
DEFAULT_CHECKPOINT = r"D:\CodeProject\ModelTrainRepo\Checkpoint\pick_pen_give_human\smolvla\checkpoints\020000\pretrained_model"
DEFAULT_DATASET = r"D:\Huggingface Cache\lerobot\CozyFire\pick_pen_give_human"


def annotate_smolvla_cache(
    cache_list: Sequence[np.ndarray], num_vlm_layers: int = 16, self_attn_every_n: int = 2,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    remaining = list(cache_list)
    for layer_idx in range(num_vlm_layers):
        if not remaining:
            break
        annotated.append({
            "layer_idx": layer_idx, "type": "self_attn", "phase": 1,
            "step": None, "probs": remaining.pop(0),
        })
    step = 0
    while remaining:
        for layer_idx in range(num_vlm_layers):
            if not remaining:
                break
            annotated.append({
                "layer_idx": layer_idx,
                "type": "self_attn" if layer_idx % self_attn_every_n == 0 else "expert_cross_attn",
                "phase": 2, "step": step, "probs": remaining.pop(0),
            })
        step += 1
    return annotated


class SmolVLAAdapter:
    name = "SmolVLA"
    default_checkpoint = DEFAULT_CHECKPOINT
    default_dataset = DEFAULT_DATASET
    restart_required_message = "get_local 会修改模型函数字节码；切换 checkpoint 请重启脚本"

    def can_reuse_context(self, context: dict[str, Any] | None, checkpoint: str) -> bool:
        return bool(context and context.get("checkpoint") == checkpoint)

    def load_context(self, checkpoint: str) -> dict[str, Any]:
        from visualizer import get_local
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        get_local.activate()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = SmolVLAPolicy.from_pretrained(checkpoint)
        policy.to(device)
        policy.eval()
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        model = policy.model.vlm_with_expert
        model.eager_attention_forward = types.MethodType(
            get_local("probs")(model.eager_attention_forward.__func__), model)
        return {
            "checkpoint": checkpoint, "policy": policy,
            "preprocessor": preprocessor, "model": model,
        }

    def capture(self, context: dict[str, Any], dataset_repo: str, frame_idx: int) -> CaptureResult:
        from visualizer import get_local

        policy, model = context["policy"], context["model"]
        item = load_lerobot_frame(policy.config, dataset_repo, frame_idx)
        image_keys, images = collect_image_observations(item)
        batch = context["preprocessor"](item)
        language_tokens = int(batch["observation.language.tokens"].shape[-1])
        get_local.clear()
        with torch.no_grad():
            policy.predict_action_chunk(batch)
        raw_cache = get_local.cache.get(CACHE_KEY, [])
        if not raw_cache:
            raise RuntimeError(f"未在 get_local.cache[{CACHE_KEY!r}] 中捕获 attention")
        cache = annotate_smolvla_cache(
            raw_cache, model.num_vlm_layers, model.self_attn_every_n_layers)
        layout = infer_token_layout(
            cache, language_tokens, len(image_keys),
            image_shapes=[image.shape[:2] for image in images],
            image_keys=image_keys,
        )
        return CaptureResult(
            cache=cache, layout=layout, images=images, image_keys=image_keys,
        )