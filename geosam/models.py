"""Model adapters for GeoSAM promptable segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

import numpy as np
import torch
from PIL import Image
from ultralytics import SAM

from geosam.logging import setup_logger

if TYPE_CHECKING:
    from ultralytics.models.sam.predict import Predictor

    from geosam.query import Points

logger = setup_logger(__name__)

ModelType = Literal["sam", "sam2", "sam3"]
ImageSource = str | Path | np.ndarray
PromptBoxes = np.ndarray | list[list[float]]
PromptPoints = np.ndarray | list[list[float]]
PromptLabels = np.ndarray | list[int]
PromptMasks = np.ndarray | list[np.ndarray]


class SAM2FeaturePayload(TypedDict):
    """Feature payload returned by SAM2-like image encoders."""

    image_embed: torch.Tensor
    high_res_feats: list[torch.Tensor]


EncodedFeaturePayload = torch.Tensor | SAM2FeaturePayload
FeatureKind = Literal["sam", "sam2"]


def _clone_feature_payload(
    features: EncodedFeaturePayload,
    device: str | torch.device | None = None,
) -> EncodedFeaturePayload:
    """Clone a feature payload and optionally move it to another device."""
    if isinstance(features, torch.Tensor):
        return (
            features.detach().clone().to(device=device)
            if device is not None
            else features.detach().clone()
        )

    if isinstance(features, dict):
        return {
            "image_embed": (
                features["image_embed"].detach().clone().to(device=device)
                if device is not None
                else features["image_embed"].detach().clone()
            ),
            "high_res_feats": [
                tensor.detach().clone().to(device=device)
                if device is not None
                else tensor.detach().clone()
                for tensor in features["high_res_feats"]
            ],
        }

    msg = f"Unsupported feature payload type: {type(features)!r}"
    logger.error(msg)
    raise TypeError(msg)


def _feature_kind(features: EncodedFeaturePayload) -> FeatureKind:
    """Return the feature kind for an encoded payload."""
    return "sam2" if isinstance(features, dict) else "sam"


def _normalize_shape(shape: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize an image-size value to ``(height, width)``."""
    if isinstance(shape, int):
        return (shape, shape)
    if len(shape) != 2:
        msg = f"Invalid image shape value: {shape!r}"
        logger.error(msg)
        raise ValueError(msg)
    return (int(shape[0]), int(shape[1]))


@dataclass(slots=True, frozen=True)
class ModelSpec:
    """Configuration describing a GeoSAM model instance.

    Parameters
    ----------
    model_type : {"sam", "sam2", "sam3"}
        Declared model family.
    checkpoint_path : str | Path
        Path to the model checkpoint.
    device : str | torch.device | None, optional
        Optional inference device.
    imgsz : int | tuple[int, int], optional
        Predictor image size.
    supports_feature_reuse : bool | None, optional
        Whether the model supports feature caching and reuse. If omitted, the
        value is inferred from ``model_type``.

    """

    model_type: ModelType
    checkpoint_path: str | Path
    device: str | torch.device | None = None
    imgsz: int | tuple[int, int] = 1024
    supports_feature_reuse: bool | None = None

    @property
    def resolved_checkpoint_path(self) -> str:
        """Return the absolute checkpoint path."""
        return str(Path(self.checkpoint_path).expanduser().resolve())

    @property
    def resolved_imgsz(self) -> tuple[int, int]:
        """Return the normalized image size."""
        return _normalize_shape(self.imgsz)

    @property
    def resolved_supports_feature_reuse(self) -> bool:
        """Return the effective feature-reuse capability flag."""
        if self.supports_feature_reuse is not None:
            return self.supports_feature_reuse
        return self.model_type in {"sam", "sam2"}


@dataclass(slots=True)
class EncodedImageFeatures:
    """Encoded image features that can be persisted and reused."""

    model_type: ModelType
    checkpoint_path: str
    src_shape: tuple[int, int]
    dst_shape: tuple[int, int]
    features: EncodedFeaturePayload

    @property
    def feature_kind(self) -> FeatureKind:
        """Return the underlying feature payload type."""
        return _feature_kind(self.features)

    def describe(self) -> dict[str, Any]:
        """Return human-readable metadata describing the encoded features."""
        if isinstance(self.features, torch.Tensor):
            structure: dict[str, Any] = {"embedding_shape": tuple(self.features.shape)}
        else:
            structure = {
                "image_embed_shape": tuple(self.features["image_embed"].shape),
                "high_res_feat_shapes": [
                    tuple(tensor.shape) for tensor in self.features["high_res_feats"]
                ],
            }

        return {
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "feature_kind": self.feature_kind,
            "src_shape": self.src_shape,
            "dst_shape": self.dst_shape,
            "structure": structure,
        }

    def save(self, file_path: str | Path) -> Path:
        """Persist encoded features to disk."""
        target_path = Path(file_path).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "src_shape": self.src_shape,
            "dst_shape": self.dst_shape,
            "feature_kind": self.feature_kind,
            "features": _clone_feature_payload(self.features, device="cpu"),
        }
        torch.save(payload, target_path)
        return target_path

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> EncodedImageFeatures:
        """Load encoded features from a file."""
        source_path = Path(file_path).expanduser().resolve()
        payload = torch.load(source_path, map_location=map_location)
        required_keys = {
            "model_type",
            "checkpoint_path",
            "src_shape",
            "dst_shape",
            "features",
        }
        if not isinstance(payload, dict) or not required_keys.issubset(payload):
            msg = f"Invalid encoded feature payload: {source_path}"
            logger.error(msg)
            raise ValueError(msg)

        return cls(
            model_type=payload["model_type"],
            checkpoint_path=str(payload["checkpoint_path"]),
            src_shape=tuple(payload["src_shape"]),
            dst_shape=tuple(payload["dst_shape"]),
            features=payload["features"],
        )


@dataclass(slots=True)
class GeoSamPrediction:
    """Model prediction output for prompt-based segmentation."""

    masks: torch.Tensor | None
    boxes: torch.Tensor

    @property
    def scores(self) -> torch.Tensor:
        """Return the score column extracted from prediction boxes."""
        if self.boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.boxes.device)
        return self.boxes[:, 4]


class GeoSamModelAdapter(Protocol):
    """Protocol implemented by all GeoSAM model adapters."""

    spec: ModelSpec

    def encode_image(self, image: ImageSource) -> EncodedImageFeatures:
        """Encode an image into reusable features."""

    def predict_image(
        self,
        image: ImageSource,
        *,
        bboxes: PromptBoxes | None = None,
        points: PromptPoints | Points | None = None,
        labels: PromptLabels | None = None,
        masks: PromptMasks | None = None,
        multimask_output: bool = False,
    ) -> GeoSamPrediction:
        """Run prompt-based prediction from an image."""

    def predict_features(
        self,
        encoded: EncodedImageFeatures,
        *,
        bboxes: PromptBoxes | None = None,
        points: PromptPoints | Points | None = None,
        labels: PromptLabels | None = None,
        masks: PromptMasks | None = None,
        multimask_output: bool = False,
        dst_shape: tuple[int, int] | None = None,
    ) -> GeoSamPrediction:
        """Run prompt-based prediction from cached features."""


class UltralyticsSamAdapter:
    """Ultralytics-backed adapter for SAM-family models."""

    def __init__(self, spec: ModelSpec) -> None:
        """Initialize an Ultralytics SAM adapter."""
        self.spec = spec
        self.model = SAM(self.spec.resolved_checkpoint_path)

    def _predictor_overrides(self) -> dict[str, Any]:
        """Build predictor overrides for Ultralytics."""
        overrides: dict[str, Any] = {
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "imgsz": self.spec.resolved_imgsz,
        }
        if self.spec.device is not None:
            overrides["device"] = str(self.spec.device)
        return overrides

    def _ensure_predictor(self) -> Predictor:
        """Create and configure the underlying Ultralytics predictor."""
        if self.model.predictor is None:
            predictor_class = self.model.task_map["segment"]["predictor"]
            predictor = predictor_class(
                overrides=self._predictor_overrides(),
                _callbacks=self.model.callbacks,
            )
            predictor.setup_model(model=self.model.model, verbose=False)
            self.model.predictor = predictor
        return self.model.predictor

    def _ensure_feature_reuse_supported(self) -> None:
        """Validate feature-reuse support for the adapter."""
        if not self.spec.resolved_supports_feature_reuse:
            msg = (
                f"Model type {self.spec.model_type!r} does not support cached "
                "feature reuse in the current adapter configuration."
            )
            logger.error(msg)
            raise NotImplementedError(msg)

    @staticmethod
    def _source_shape_from_image(image: ImageSource) -> tuple[int, int]:
        """Extract source image shape as ``(height, width)``."""
        if isinstance(image, np.ndarray):
            if image.ndim < 2:
                msg = (
                    f"Image array must have at least two dimensions. Got {image.shape}."
                )
                logger.error(msg)
                raise ValueError(msg)
            return (int(image.shape[0]), int(image.shape[1]))

        image_path = Path(image).expanduser().resolve()
        if not image_path.exists():
            msg = f"Image path does not exist: {image_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        with Image.open(image_path) as source_image:
            width, height = source_image.size
        return (int(height), int(width))

    def _validate_checkpoint_match(self, encoded: EncodedImageFeatures) -> None:
        """Ensure loaded features match the adapter checkpoint and model type."""
        current_name = Path(self.spec.resolved_checkpoint_path).name
        encoded_name = Path(encoded.checkpoint_path).name
        if current_name != encoded_name:
            msg = (
                f"Checkpoint mismatch: adapter uses {current_name!r}, but features "
                f"were created with {encoded_name!r}."
            )
            logger.error(msg)
            raise ValueError(msg)

        if encoded.model_type != self.spec.model_type:
            msg = (
                f"Model type mismatch: adapter uses {self.spec.model_type!r}, but "
                f"features were created with {encoded.model_type!r}."
            )
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def _normalize_point_prompts(
        points: PromptPoints | Points | None,
        labels: PromptLabels | None,
    ) -> tuple[PromptPoints | None, PromptLabels | None]:
        """Normalize point prompts into coordinates and labels."""
        if points is None:
            return None, labels

        from geosam.query.points import Points as QueryPoints

        if isinstance(points, QueryPoints):
            if labels is not None:
                msg = (
                    "Explicit labels cannot be combined with a Points object. "
                    "Attach labels directly to Points."
                )
                logger.error(msg)
                raise ValueError(msg)
            return points.to_sam_prompt()

        return points, labels

    def encode_image(self, image: ImageSource) -> EncodedImageFeatures:
        """Encode an image into reusable SAM features."""
        self._ensure_feature_reuse_supported()
        predictor = self._ensure_predictor()
        predictor.set_image(str(image) if isinstance(image, Path) else image)
        if predictor.features is None:
            msg = f"Feature extraction failed for image source {image!r}."
            logger.error(msg)
            raise RuntimeError(msg)

        return EncodedImageFeatures(
            model_type=self.spec.model_type,
            checkpoint_path=self.spec.resolved_checkpoint_path,
            src_shape=self._source_shape_from_image(image),
            dst_shape=_normalize_shape(predictor.args.imgsz),
            features=_clone_feature_payload(predictor.features, device="cpu"),
        )

    def predict_features(
        self,
        encoded: EncodedImageFeatures,
        *,
        bboxes: PromptBoxes | None = None,
        points: PromptPoints | Points | None = None,
        labels: PromptLabels | None = None,
        masks: PromptMasks | None = None,
        multimask_output: bool = False,
        dst_shape: tuple[int, int] | None = None,
    ) -> GeoSamPrediction:
        """Run promptable inference from cached image features."""
        self._ensure_feature_reuse_supported()
        self._validate_checkpoint_match(encoded)

        normalized_points, normalized_labels = self._normalize_point_prompts(
            points,
            labels,
        )
        if normalized_points is not None and normalized_labels is None:
            msg = "Point prompts require matching labels."
            logger.error(msg)
            raise ValueError(msg)

        predictor = self._ensure_predictor()
        predictor_features = _clone_feature_payload(
            encoded.features,
            device=predictor.device,
        )
        pred_masks, pred_boxes = predictor.inference_features(
            features=predictor_features,
            src_shape=encoded.src_shape,
            dst_shape=dst_shape or encoded.dst_shape,
            bboxes=bboxes,
            points=normalized_points,
            labels=normalized_labels,
            masks=masks,
            multimask_output=multimask_output,
        )
        return GeoSamPrediction(masks=pred_masks, boxes=pred_boxes)

    def predict_image(
        self,
        image: ImageSource,
        *,
        bboxes: PromptBoxes | None = None,
        points: PromptPoints | Points | None = None,
        labels: PromptLabels | None = None,
        masks: PromptMasks | None = None,
        multimask_output: bool = False,
    ) -> GeoSamPrediction:
        """Run promptable inference directly from an image."""
        encoded = self.encode_image(image)
        return self.predict_features(
            encoded,
            bboxes=bboxes,
            points=points,
            labels=labels,
            masks=masks,
            multimask_output=multimask_output,
        )


class Sam3ModelAdapter(UltralyticsSamAdapter):
    """Adapter entry reserved for SAM3-style checkpoints."""

    def _ensure_feature_reuse_supported(self) -> None:
        """Require explicit opt-in for SAM3 feature reuse."""
        if not self.spec.resolved_supports_feature_reuse:
            msg = (
                "SAM3 feature reuse is not enabled by default. Set "
                "supports_feature_reuse=True in ModelSpec after validating the "
                "checkpoint and Ultralytics runtime support."
            )
            logger.error(msg)
            raise NotImplementedError(msg)


def build_model_adapter(spec: ModelSpec) -> GeoSamModelAdapter:
    """Build a model adapter from a model specification."""
    if spec.model_type in {"sam", "sam2"}:
        return UltralyticsSamAdapter(spec)
    if spec.model_type == "sam3":
        return Sam3ModelAdapter(spec)

    msg = f"Unsupported model type: {spec.model_type!r}"
    logger.error(msg)
    raise ValueError(msg)
