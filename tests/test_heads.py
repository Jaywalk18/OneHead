import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eml import PROBE_HEADS, build_projection_head  # noqa: E402


class ProjectionHeadTests(unittest.TestCase):
    def test_all_stress_heads_are_finite_and_shape_preserving(self):
        features = torch.randn(8, 32)
        for kind in PROBE_HEADS:
            with self.subTest(kind=kind):
                head = build_projection_head(kind, in_dim=32, hidden_dim=16, out_dim=8)
                output = head(features)
                self.assertEqual(tuple(output.shape), (8, 8))
                self.assertTrue(torch.isfinite(output).all())

    def test_fixed_norm_control_sets_each_sample_norm(self):
        features = torch.randn(8, 32)
        head = build_projection_head(
            "scalenorm", in_dim=32, hidden_dim=16, out_dim=8, power=4.0
        )
        norms = head(features).norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.full_like(norms, 4.0), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
