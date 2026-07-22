import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from custom_visualizer.common import DisplayItem, serialize_display_item
from custom_visualizer.policies.smolvla_apt import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    SETTINGS_PATH_APT,
    SmolVLAAptAdapter,
    build_apt_token_layout,
    collect_apt_source_images,
    make_apt_cache_entry,
    reconstruct_action_attention_probs,
    validate_stage1,
)


class FakeAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=8, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)


class SmolVLAAptAdapterTests(unittest.TestCase):
    def test_adapter_defaults_and_independent_settings(self):
        adapter = SmolVLAAptAdapter()
        self.assertEqual(adapter.default_checkpoint, DEFAULT_CHECKPOINT)
        self.assertEqual(
            DEFAULT_CHECKPOINT,
            r"D:\CodeProject\ModelTrainRepo\Checkpoint\smolvla_apt\stage_1"
            r"\Cozy_pick_pen_give_human\checkpoints\001000\pretrained_model",
        )
        self.assertEqual(adapter.default_dataset, DEFAULT_DATASET)
        self.assertEqual(adapter.settings_path, SETTINGS_PATH_APT)
        self.assertTrue(adapter.can_reuse_context({"checkpoint": "a"}, "a"))
        self.assertFalse(adapter.can_reuse_context({"checkpoint": "a"}, "b"))

    def test_stage1_validation_rejects_stage0(self):
        validate_stage1(SimpleNamespace(train_stage=1))
        with self.assertRaisesRegex(ValueError, "仅支持 Stage 1"):
            validate_stage1(SimpleNamespace(train_stage=0))

    def test_even_and_odd_layers_use_apt_semantic_labels(self):
        probs = np.zeros((1, 2, 3, 4), dtype=np.float32)
        even = make_apt_cache_entry(0, 0, probs)
        odd = make_apt_cache_entry(0, 1, probs)
        self.assertEqual(
            (even["type"], even["label"], even["attention_kind"], even["mask_type"]),
            ("vla_likelihood", "VLA似然", "self_attn", "full"),
        )
        self.assertEqual(
            (odd["type"], odd["label"], odd["attention_kind"], odd["mask_type"]),
            ("va_prior", "VA先验", "self_attn", "dilated"),
        )
        payload = serialize_display_item(DisplayItem(even, 0))
        self.assertEqual(payload["title"], "S0 · L0 · H0 · VLA似然")
        self.assertEqual(payload["attention_kind"], "self_attn")

    def test_reconstructed_probs_match_sdpa_for_action_queries(self):
        torch.manual_seed(3)
        layer = FakeAttentionBlock()
        x = torch.randn(1, 6, 8)
        mask = torch.ones(1, 6, 6, dtype=torch.bool)
        mask[:, -2:, 2] = False
        probs = reconstruct_action_attention_probs(
            layer, x, mask, position_ids=None, action_tokens=2, apply_rope_fn=None
        )
        self.assertEqual(tuple(probs.shape), (1, 2, 2, 6))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(1, 2, 2), atol=1e-6))
        self.assertTrue(torch.equal(probs[..., 2], torch.zeros_like(probs[..., 2])))

        normalized = layer.norm1(x)
        q = layer.q_proj(normalized).view(1, 6, 2, 4).transpose(1, 2)
        k = layer.k_proj(normalized).view(1, 6, 2, 4).transpose(1, 2)
        v = layer.v_proj(normalized).view(1, 6, 2, 4).transpose(1, 2)
        expected = F.scaled_dot_product_attention(q[:, :, -2:], k, v, attn_mask=mask[:, None, -2:])
        actual = probs.to(v.dtype) @ v
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_fully_masked_action_row_gets_model_self_loop(self):
        layer = FakeAttentionBlock()
        x = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, 4, dtype=torch.bool)
        mask[:, -1, :] = False
        probs = reconstruct_action_attention_probs(
            layer, x, mask, position_ids=None, action_tokens=1, apply_rope_fn=None
        )
        expected = torch.zeros_like(probs)
        expected[..., -1] = 1.0
        self.assertTrue(torch.equal(probs, expected))

    def test_source_images_follow_camera_order_and_include_missing_placeholder(self):
        config = SimpleNamespace(
            image_features={
                "observation.images.front": object(),
                "observation.images.wrist": object(),
                "observation.images.side": object(),
            },
            camera_order=[
                "observation.images.wrist",
                "observation.images.front",
                "observation.images.side",
            ],
            resize_imgs_with_padding=(6, 10),
        )
        item = {
            "observation.images.front": np.zeros((3, 8, 8), dtype=np.float32),
            "observation.images.wrist": np.ones((3, 8, 8), dtype=np.float32),
        }
        keys, images = collect_apt_source_images(item, config)
        self.assertEqual(keys, config.camera_order)
        self.assertEqual(len(images), 3)
        self.assertEqual(images[2].shape, (6, 10, 3))
        self.assertTrue(np.all(images[2] == 0))

    def test_layout_skips_image_special_tokens_for_overlay(self):
        probs = np.zeros((1, 2, 3, 18), dtype=np.float32)
        cache = [make_apt_cache_entry(0, 0, probs)]
        images = [
            np.zeros((8, 8, 3), dtype=np.float32),
            np.zeros((4, 8, 3), dtype=np.float32),
        ]
        layout = build_apt_token_layout(
            cache=cache,
            patch_counts=[4, 4],
            image_keys=["observation.images.front", "observation.images.wrist"],
            images=images,
            language_tokens=2,
            add_image_special_tokens=True,
        )
        # 2 views * (start + 4 patches + end) = 12 leading image-region tokens.
        self.assertEqual(layout.image_tokens, 12)
        self.assertEqual(layout.language_tokens, 2)
        self.assertEqual(layout.state_tokens, 1)
        self.assertEqual(layout.prefix_tokens, 15)
        self.assertEqual(layout.image_layouts[0].token_start, 1)
        self.assertEqual(layout.image_layouts[1].token_start, 7)
        self.assertEqual(layout.image_layouts[0].token_count, 4)
        self.assertIsNone(layout.overlay_reason)


if __name__ == "__main__":
    unittest.main()
