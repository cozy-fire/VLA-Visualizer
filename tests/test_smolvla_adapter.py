import unittest

import numpy as np

from custom_visualizer.policies.smolvla import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    SmolVLAAdapter,
    annotate_smolvla_cache,
)


class SmolVLAAdapterTests(unittest.TestCase):
    def test_annotate_smolvla_cache_marks_phase_step_layer_and_type(self):
        raw = [np.zeros((1, 2, 3, 4), dtype=np.float32) for _ in range(8)]
        cache = annotate_smolvla_cache(raw, num_vlm_layers=4, self_attn_every_n=2)

        self.assertEqual([entry["phase"] for entry in cache[:4]], [1, 1, 1, 1])
        self.assertEqual([entry["step"] for entry in cache[:4]], [None, None, None, None])
        self.assertEqual([entry["layer_idx"] for entry in cache[:4]], [0, 1, 2, 3])

        phase2 = cache[4:]
        self.assertEqual([entry["phase"] for entry in phase2], [2, 2, 2, 2])
        self.assertEqual([entry["step"] for entry in phase2], [0, 0, 0, 0])
        self.assertEqual([entry["layer_idx"] for entry in phase2], [0, 1, 2, 3])
        self.assertEqual(
            [entry["type"] for entry in phase2],
            ["self_attn", "expert_cross_attn", "self_attn", "expert_cross_attn"],
        )

    def test_adapter_defaults_and_context_reuse(self):
        adapter = SmolVLAAdapter()
        self.assertEqual(adapter.default_checkpoint, DEFAULT_CHECKPOINT)
        self.assertEqual(adapter.default_dataset, DEFAULT_DATASET)
        self.assertTrue(adapter.can_reuse_context({"checkpoint": "a"}, "a"))
        self.assertFalse(adapter.can_reuse_context({"checkpoint": "a"}, "b"))
        self.assertFalse(adapter.can_reuse_context(None, "a"))


if __name__ == "__main__":
    unittest.main()