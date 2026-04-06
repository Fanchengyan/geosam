"""Compatibility wrapper around the GeoSAM model adapter layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from geosam.models import (
    EncodedImageFeatures,
    GeoSamPrediction,
    ImageSource,
    ModelSpec,
    ModelType,
    PromptBoxes,
    PromptLabels,
    PromptMasks,
    PromptPoints,
    build_model_adapter,
)

if TYPE_CHECKING:
    from pathlib import Path

    import torch


class SAMFeatureEncoder:
    """Encode, persist, reload, and reuse SAM image features.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a model checkpoint.
    imgsz : int | tuple[int, int], optional
        Predictor image size.
    device : str | torch.device | None, optional
        Optional inference device.
    model_type : {"sam", "sam2", "sam3"}, optional
        Declared model family.
    supports_feature_reuse : bool | None, optional
        Optional override for model feature-reuse support.

    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        imgsz: int | tuple[int, int] = 1024,
        device: str | torch.device | None = None,
        *,
        model_type: ModelType = "sam",
        supports_feature_reuse: bool | None = None,
    ) -> None:
        """Initialize a SAM feature encoder wrapper."""
        self.spec = ModelSpec(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            device=device,
            imgsz=imgsz,
            supports_feature_reuse=supports_feature_reuse,
        )
        self.adapter = build_model_adapter(self.spec)

    def encode(self, image: ImageSource) -> EncodedImageFeatures:
        """Encode an image into reusable features."""
        return self.adapter.encode_image(image)

    def encode_to_file(
        self,
        image: ImageSource,
        file_path: str | Path,
    ) -> EncodedImageFeatures:
        """Encode an image and persist the resulting features."""
        encoded = self.encode(image)
        encoded.save(file_path)
        return encoded

    def load_features(
        self,
        file_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> EncodedImageFeatures:
        """Load encoded features from a file."""
        return EncodedImageFeatures.load(file_path, map_location=map_location)

    def inference_features(
        self,
        encoded: EncodedImageFeatures,
        *,
        bboxes: PromptBoxes | None = None,
        points: PromptPoints | object | None = None,
        labels: PromptLabels | None = None,
        masks: PromptMasks | None = None,
        multimask_output: bool = False,
        dst_shape: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run promptable inference from cached features."""
        prediction = self.adapter.predict_features(
            encoded,
            bboxes=bboxes,
            points=points,
            labels=labels,
            masks=masks,
            multimask_output=multimask_output,
            dst_shape=dst_shape,
        )
        return prediction.masks, prediction.boxes

    def inference_feature_file(
        self,
        file_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        bboxes: PromptBoxes | None = None,
        points: PromptPoints | object | None = None,
        labels: PromptLabels | None = None,
        masks: PromptMasks | None = None,
        multimask_output: bool = False,
        dst_shape: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Load cached features and run promptable inference."""
        encoded = self.load_features(file_path, map_location=map_location)
        return self.inference_features(
            encoded,
            bboxes=bboxes,
            points=points,
            labels=labels,
            masks=masks,
            multimask_output=multimask_output,
            dst_shape=dst_shape,
        )

    def predict_image(
        self,
        image: ImageSource,
        *,
        bboxes: PromptBoxes | None = None,
        points: PromptPoints | object | None = None,
        labels: PromptLabels | None = None,
        masks: PromptMasks | None = None,
        multimask_output: bool = False,
    ) -> GeoSamPrediction:
        """Run promptable inference directly from an image."""
        return self.adapter.predict_image(
            image,
            bboxes=bboxes,
            points=points,
            labels=labels,
            masks=masks,
            multimask_output=multimask_output,
        )
