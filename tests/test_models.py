from __future__ import annotations

from typing import Optional, Union
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geosam.models import ModelSpec, Sam3ModelAdapter, build_model_adapter
from geosam.runtime import create_model_spec_from_checkpoint


class FakePredictor:
    def __init__(
        self,
        overrides: Optional[dict[str, object]] = None,
        _callbacks: Optional[dict[str, object]] = None,
    ) -> None:
        overrides = overrides or {}
        self.args = SimpleNamespace(imgsz=overrides.get("imgsz", (1024, 1024)))
        self.device = torch.device("cpu")
        self.features: Optional[dict[str, Union[torch.Tensor, list[torch.Tensor]]]] = (
            None
        )
        self.last_inference_kwargs: Optional[dict[str, object]] = None

    def setup_model(self, model: object, verbose: bool = False) -> None:
        self.model = model

    def set_image(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        self.args.imgsz = (height, width)
        self.features = {
            "image_embed": torch.ones((1, 1, 1, 1), dtype=torch.float32),
            "high_res_feats": [torch.ones((1, 1, 1, 1), dtype=torch.float32)],
        }

    def inference_features(
        self,
        *,
        features: dict[str, Union[torch.Tensor, list[torch.Tensor]]],
        src_shape: tuple[int, int],
        dst_shape: tuple[int, int],
        bboxes: object = None,
        points: object = None,
        labels: object = None,
        masks: object = None,
        multimask_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_inference_kwargs = {
            "features": features,
            "src_shape": src_shape,
            "dst_shape": dst_shape,
            "bboxes": bboxes,
            "points": points,
            "labels": labels,
            "masks": masks,
            "multimask_output": multimask_output,
        }
        pred_masks = torch.ones((1, src_shape[0], src_shape[1]), dtype=torch.bool)
        pred_boxes = torch.tensor(
            [[0.0, 0.0, float(src_shape[1]), float(src_shape[0]), 0.9, 0.0]],
            dtype=torch.float32,
        )
        return pred_masks, pred_boxes


class FakeSAM:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.callbacks: dict[str, object] = {}
        self.model = object()
        self.predictor: Optional[FakePredictor] = None
        self.task_map = {"segment": {"predictor": FakePredictor}}


def test_sam3_predict_image_without_feature_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("geosam.models.SAM", FakeSAM)

    model_spec = ModelSpec(
        "sam3",
        checkpoint_path="sam3.pt",
        supports_feature_reuse=False,
    )
    adapter = build_model_adapter(model_spec)
    image = np.zeros((6, 8, 3), dtype=np.uint8)

    prediction = adapter.predict_image(
        image,
        bboxes=[[1.0, 1.0, 4.0, 5.0]],
        multimask_output=True,
    )

    assert isinstance(adapter, Sam3ModelAdapter)
    assert prediction.masks is not None
    assert prediction.masks.shape == (1, 6, 8)
    assert prediction.boxes.shape == (1, 6)
    assert adapter.model.predictor is not None
    assert adapter.model.predictor.last_inference_kwargs is not None
    assert adapter.model.predictor.last_inference_kwargs["bboxes"] == [
        [1.0, 1.0, 4.0, 5.0]
    ]

    with pytest.raises(NotImplementedError, match="SAM3 feature reuse"):
        adapter.encode_image(image)


def test_sam3_feature_reuse_defaults_follow_checkpoint_name() -> None:
    sam3_spec = ModelSpec("sam3", checkpoint_path="sam3.pt")
    multiplex_spec = ModelSpec("sam3", checkpoint_path="sam3.1_multiplex.pt")
    runtime_spec = create_model_spec_from_checkpoint("sam3.pt")
    runtime_multiplex_spec = create_model_spec_from_checkpoint("sam3.1_multiplex.pt")

    assert sam3_spec.resolved_supports_feature_reuse is True
    assert multiplex_spec.resolved_supports_feature_reuse is True
    assert runtime_spec.resolved_supports_feature_reuse is True
    assert runtime_multiplex_spec.resolved_supports_feature_reuse is True
