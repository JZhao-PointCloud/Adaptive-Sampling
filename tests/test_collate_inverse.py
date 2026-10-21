import unittest
import importlib.util
from pathlib import Path
import torch

_UTILS_PATH = (
    Path(__file__).resolve().parents[1] / "pointcept" / "datasets" / "utils.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "pointcept_dataset_utils_for_test", _UTILS_PATH
)
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)
collate_fn = _UTILS.collate_fn


class CollateInverseTest(unittest.TestCase):
    def test_inverse_is_shifted_to_batched_sparse_logits(self):
        batch = [
            dict(
                coord=torch.tensor([[0.0], [1.0]]),
                segment=torch.tensor([0, 1]),
                origin_segment=torch.tensor([0, 1, 1]),
                inverse=torch.tensor([0, 1, 1]),
            ),
            dict(
                coord=torch.tensor([[10.0], [11.0], [12.0]]),
                segment=torch.tensor([2, 3, 4]),
                origin_segment=torch.tensor([2, 4, 3, 4]),
                inverse=torch.tensor([0, 2, 1, 2]),
            ),
        ]
        out = collate_fn(batch)
        self.assertTrue(
            torch.equal(out["inverse"], torch.tensor([0, 1, 1, 2, 4, 3, 4]))
        )
        sparse_pred = torch.arange(out["coord"].shape[0])
        self.assertTrue(
            torch.equal(
                sparse_pred[out["inverse"]], torch.tensor([0, 1, 1, 2, 4, 3, 4])
            )
        )

    def test_single_sample_inverse_is_unchanged(self):
        batch = [
            dict(
                coord=torch.tensor([[0.0], [1.0]]),
                segment=torch.tensor([0, 1]),
                origin_segment=torch.tensor([0, 1, 1]),
                inverse=torch.tensor([0, 1, 1]),
            )
        ]
        out = collate_fn(batch)
        self.assertTrue(torch.equal(out["inverse"], torch.tensor([0, 1, 1])))


if __name__ == "__main__":
    unittest.main()
