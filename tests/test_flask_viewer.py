import base64
import unittest
from unittest.mock import patch

import numpy as np

from custom_visualizer.common import CaptureResult, infer_token_layout
from custom_visualizer.web import create_app


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


class FakeAdapter:
    name = "FakePolicy"
    default_checkpoint = "checkpoint-default"
    default_dataset = "dataset-default"
    restart_required_message = "restart required"

    def can_reuse_context(self, context, checkpoint):
        return bool(context and context.get("checkpoint") == checkpoint)

    def load_context(self, checkpoint):
        return {"checkpoint": checkpoint}

    def capture(self, context, dataset_repo, frame_idx):
        cache = fake_cache(image_tokens=4)
        layout = infer_token_layout(
            cache, language_tokens=2, image_shapes=[(8, 8)], image_keys=["observation.images.front"])
        cache[0]["probs"][0, 0, :, :4] = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.float32)
        cache[1]["probs"][0, 0, :, :4] = np.array([[3, 2, 1, 0], [4, 3, 2, 1]], dtype=np.float32)
        return CaptureResult(
            cache=cache,
            layout=layout,
            images=[np.zeros((8, 8, 3), dtype=np.float32)],
            image_keys=["observation.images.front"],
        )


class FlaskViewerTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(FakeAdapter(), persist_settings=False)
        self.client = self.app.test_client()

    def capture(self):
        response = self.client.post('/api/capture', json={
            "checkpoint": "checkpoint-default",
            "dataset": "dataset-default",
            "frame_idx": 0,
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def test_index_and_settings(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        settings = self.client.get('/api/settings').get_json()
        self.assertEqual(settings["adapter"], "FakePolicy")
        self.assertEqual(settings["checkpoint"], "checkpoint-default")
        self.assertEqual(settings["dataset"], "dataset-default")
        self.assertEqual(settings["checkpoint_history"], ["checkpoint-default"])
        self.assertEqual(settings["dataset_history"], ["dataset-default"])

    def test_capture_then_static_items_return_raw_payload(self):
        summary = self.capture()
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["steps"], [0, 1, 2])
        self.assertEqual(summary["layers"], [0, 1])
        self.assertEqual(summary["heads"], [0, 1, 2])
        self.assertIn("layout", summary)
        self.assertEqual(summary["image_keys"], ["observation.images.front"])
        self.assertTrue(summary["source_images"][0]["src"].startswith("data:image/png;base64,"))

        items = self.client.get('/api/items?page=0&overlay=0').get_json()
        self.assertTrue(items["ok"])
        self.assertEqual(items["total"], 18)
        self.assertEqual(len(items["items"]), 12)
        first = items["items"][0]
        self.assertNotIn("image", first)
        self.assertEqual(first["probs"]["dtype"], "float32")
        raw = base64.b64decode(first["probs"]["data"])
        restored = np.frombuffer(raw, dtype=np.float32).reshape(first["probs"]["shape"])
        self.assertEqual(restored.shape, (2, 9))

    def test_overlay_does_not_change_item_count(self):
        self.capture()
        matrix = self.client.get('/api/items?page=0&overlay=0').get_json()
        overlay = self.client.get('/api/items?page=0&overlay=1').get_json()
        self.assertEqual(matrix["total"], overlay["total"])
        self.assertEqual(len(matrix["items"]), len(overlay["items"]))
        self.assertIn("probs", overlay["items"][0])

    def test_all_six_dynamic_combinations_use_dynamic_bundle(self):
        self.capture()
        cases = [
            ("steps", "layer=0", 3, 3),
            ("steps", "head=0", 3, 2),
            ("layers", "step=0", 2, 3),
            ("layers", "head=0", 2, 3),
            ("heads", "step=0", 3, 2),
            ("heads", "layer=0", 3, 3),
        ]
        for mode, fixed_query, frame_count, items_per_frame in cases:
            with self.subTest(mode=mode, fixed_query=fixed_query):
                bundle = self.client.get(f'/api/dynamic?mode={mode}&{fixed_query}').get_json()
                self.assertTrue(bundle["ok"])
                self.assertEqual(bundle["frame_count"], frame_count)
                self.assertEqual(len(bundle["frames"]), frame_count)
                self.assertEqual(len(bundle["frames"][0]["items"]), items_per_frame)
                self.assertIn("probs", bundle["frames"][0]["items"][0])

    def test_frame_preview_endpoint_uses_paged_loader(self):
        fake_page = {
            "num_frames": 2,
            "page": 0,
            "page_size": 24,
            "total_pages": 1,
            "frames": [{
                "frame_idx": 0,
                "image_keys": ["observation.images.front"],
                "images": [{"height": 8, "width": 8, "src": "data:image/png;base64,abc"}],
                "state_summary": [{"key": "observation.state", "shape": [2], "dtype": "float32", "preview": [1, 2]}],
            }],
        }
        with patch("custom_visualizer.web.app.load_lerobot_frame_page", return_value=fake_page) as mocked:
            response = self.client.get('/api/dataset/frames?dataset=dataset-default&page=0&page_size=24&stride=16')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["frames"][0]["frame_idx"], 0)
        mocked.assert_called_once_with("dataset-default", page=0, page_size=24, stride=16)


if __name__ == "__main__":
    unittest.main()