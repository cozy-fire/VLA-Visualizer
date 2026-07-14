import unittest
from pathlib import Path

import numpy as np

from custom_visualizer.common import (
    build_dynamic_frames,
    choose_patch_grid,
    collect_image_observations,
    compose_overlay_image,
    extract_patch_scores,
    extract_view_patch_scores,
    filter_display_items,
    infer_token_layout,
    load_settings,
    patch_scores_to_grid,
    save_settings,
    static_window,
)


def fake_cache(image_tokens=4, language_tokens=2, state_tokens=1, action_tokens=2):
    prefix_tokens = image_tokens + language_tokens + state_tokens
    result = []
    for step in range(3):
        result.append({
            "phase": 2, "step": step, "layer_idx": 0, "type": "self_attn",
            "probs": np.zeros((1, 3, action_tokens, prefix_tokens + action_tokens), dtype=np.float32),
        })
        result.append({
            "phase": 2, "step": step, "layer_idx": 1, "type": "expert_cross_attn",
            "probs": np.zeros((1, 3, action_tokens, prefix_tokens), dtype=np.float32),
        })
    return result


class AttentionViewerCommonTests(unittest.TestCase):
    def test_settings_defaults_roundtrip_and_malformed_fallback(self):
        path = Path(r"C:\tmp\vlm_visualizer_settings_test.json")
        temporary = path.with_suffix(path.suffix + ".tmp")
        path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        try:
            self.assertEqual(load_settings("checkpoint-default", "dataset-default", path), {
                "checkpoint": "checkpoint-default",
                "dataset": "dataset-default",
            })
            save_settings("checkpoint-a", "dataset-a", path)
            self.assertEqual(load_settings("checkpoint-default", "dataset-default", path), {
                "checkpoint": "checkpoint-a",
                "dataset": "dataset-a",
            })
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(
                load_settings("checkpoint-default", "dataset-default", path)["checkpoint"],
                "checkpoint-default",
            )
        finally:
            path.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)

    def test_static_window_only_returns_visible_maps(self):
        items = filter_display_items(fake_cache()) * 134
        row, total_rows, visible = static_window(items, start_row=0)
        self.assertEqual(row, 0)
        self.assertGreater(total_rows, 600)
        self.assertEqual(len(visible), 12)
        last_row, _, last_visible = static_window(items, start_row=99999)
        self.assertEqual(last_row, total_rows - 3)
        self.assertLessEqual(len(last_visible), 12)

    def test_static_filter_and_expansion(self):
        cache = fake_cache()
        self.assertEqual(len(filter_display_items(cache)), 18)
        items = filter_display_items(cache, step=1, layer=0, head=2)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].head_idx, 2)

    def test_all_six_dynamic_combinations(self):
        cache = fake_cache()
        cases = [
            ("steps", dict(layer=0), 3, 3),
            ("steps", dict(head=0), 3, 2),
            ("layers", dict(step=0), 2, 3),
            ("layers", dict(head=0), 2, 3),
            ("heads", dict(step=0), 3, 2),
            ("heads", dict(layer=0), 3, 3),
        ]
        for mode, kwargs, frame_count, items_per_frame in cases:
            with self.subTest(mode=mode, kwargs=kwargs):
                frames = build_dynamic_frames(cache, mode, **kwargs)
                self.assertEqual(len(frames), frame_count)
                self.assertTrue(all(len(items) == items_per_frame for _, items in frames))

    def test_dynamic_requires_exactly_one_fixed_dimension(self):
        cache = fake_cache()
        with self.assertRaises(ValueError):
            build_dynamic_frames(cache, "steps")
        with self.assertRaises(ValueError):
            build_dynamic_frames(cache, "steps", layer=0, head=0)

    def test_collect_image_observations_uses_observation_images_prefix(self):
        item = {
            "observation.image.front": np.zeros((3, 8, 8), dtype=np.float32),
            "observation.image_timestamp": np.array([123], dtype=np.int64),
            "observation.images": {
                "wrist": np.ones((8, 8, 3), dtype=np.float32),
                "bad_scalar": np.array([1], dtype=np.float32),
            },
        }
        keys, images = collect_image_observations(item)
        self.assertEqual(keys, ["observation.images.wrist"])
        self.assertEqual([image.shape for image in images], [(8, 8, 3)])

    def test_single_view_layout_and_action_pooling_ignore_non_image_keys(self):
        cache = fake_cache(image_tokens=4)
        layout = infer_token_layout(
            cache, language_tokens=2, image_shapes=[(32, 32)], image_keys=["cam0"])
        self.assertEqual(layout.image_tokens, 4)
        self.assertIsNone(layout.overlay_reason)
        self.assertEqual(layout.grid_side, 2)
        self.assertEqual(layout.image_layouts[0].image_key, "cam0")

        entry = cache[0]
        entry["probs"][0, 1, 0, :4] = [1, 2, 3, 4]
        entry["probs"][0, 1, 1, :4] = [3, 4, 5, 6]
        entry["probs"][0, 1, :, 4:] = 1000
        np.testing.assert_allclose(extract_patch_scores(entry, 1, layout), [2, 3, 4, 5])
        np.testing.assert_allclose(
            patch_scores_to_grid(np.array([2, 3, 4, 5]), layout),
            [[0, 1/3], [2/3, 1]],
        )

    def test_multi_view_layout_splits_image_tokens_by_observation(self):
        cache = fake_cache(image_tokens=8)
        layout = infer_token_layout(
            cache, language_tokens=2, image_observation_count=2,
            image_shapes=[(32, 32), (32, 32)], image_keys=["front", "wrist"],
        )
        self.assertIsNone(layout.overlay_reason)
        self.assertEqual(len(layout.image_layouts), 2)
        self.assertEqual(layout.image_layouts[0].token_start, 0)
        self.assertEqual(layout.image_layouts[1].token_start, 4)
        self.assertEqual((layout.image_layouts[0].grid_height, layout.image_layouts[0].grid_width), (2, 2))
        self.assertEqual((layout.image_layouts[1].grid_height, layout.image_layouts[1].grid_width), (2, 2))

        entry = cache[1]
        entry["probs"][0, 0, 0, :8] = [1, 2, 3, 4, 10, 20, 30, 40]
        entry["probs"][0, 0, 1, :8] = [3, 4, 5, 6, 30, 40, 50, 60]
        np.testing.assert_allclose(
            extract_view_patch_scores(entry, 0, layout, layout.image_layouts[0]),
            [2, 3, 4, 5],
        )
        np.testing.assert_allclose(
            extract_view_patch_scores(entry, 0, layout, layout.image_layouts[1]),
            [20, 30, 40, 50],
        )

    def test_non_square_grid_uses_source_aspect_ratio(self):
        self.assertEqual(choose_patch_grid(12, image_height=32, image_width=48), (3, 4))
        self.assertEqual(choose_patch_grid(12, image_height=48, image_width=32), (4, 3))
        cache = fake_cache(image_tokens=12)
        layout = infer_token_layout(cache, language_tokens=2, image_shapes=[(32, 48)])
        image_layout = layout.image_layouts[0]
        self.assertEqual((image_layout.grid_height, image_layout.grid_width), (3, 4))

    def test_unbalanced_multi_view_tokens_disable_overlay(self):
        layout = infer_token_layout(
            fake_cache(image_tokens=5), language_tokens=2,
            image_observation_count=2, image_shapes=[(32, 32), (32, 32)],
        )
        self.assertIsNotNone(layout.overlay_reason)
        self.assertFalse(layout.image_layouts)

    def test_multi_view_overlay_composes_one_subplot_image(self):
        cache = fake_cache(image_tokens=8)
        layout = infer_token_layout(
            cache, language_tokens=2, image_observation_count=2,
            image_shapes=[(4, 4), (4, 4)], image_keys=["front", "wrist"],
        )
        entry = cache[1]
        entry["probs"][0, 0, :, :8] = np.array([
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 0, 1, 1, 2, 2, 3, 3],
        ], dtype=np.float32)
        images = [
            np.zeros((4, 4, 3), dtype=np.float32),
            np.ones((4, 4, 3), dtype=np.float32) * 0.25,
        ]
        composite = compose_overlay_image(images, entry, 0, layout)
        self.assertEqual(composite.shape, (4, 8, 3))
        self.assertGreaterEqual(float(composite.min()), 0.0)
        self.assertLessEqual(float(composite.max()), 1.0)


if __name__ == "__main__":
    unittest.main()