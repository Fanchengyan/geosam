"""Utilities for encoding, caching, and reusing Ultralytics SAM image features.

This module provides a small wrapper around the Ultralytics SAM predictor API.
It supports:

- Encoding image features with a SAM or SAM2 checkpoint
- Saving the encoded features to a local file
- Loading encoded features from a local file
- Reusing the loaded features with ``predictor.inference_features()``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import torch
from PIL import Image
from ultralytics import SAM
from ultralytics.models.sam.predict import Predictor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from geosam.query.points import Points

ImageSource = Union[str, Path, np.ndarray]
PromptBoxes = Union[np.ndarray, List[List[float]]]
PromptPoints = Union[np.ndarray, List[List[float]]]
PromptLabels = Union[np.ndarray, List[int]]
PromptMasks = Union[np.ndarray, List[np.ndarray]]


class SAM2FeaturePayload(TypedDict):
    """Typed SAM2 feature payload."""

    image_embed: torch.Tensor
    high_res_feats: List[torch.Tensor]


EncodedFeaturePayload = Union[torch.Tensor, SAM2FeaturePayload]
FeatureKind = Literal["sam", "sam2"]


def _clone_feature_payload(
    features: EncodedFeaturePayload,
    device: Optional[Union[str, torch.device]] = None,
) -> EncodedFeaturePayload:
    """Clone a feature payload and move it to the requested device.

    Parameters
    ----------
    features : EncodedFeaturePayload
        Feature payload produced by a SAM or SAM2 image encoder.
    device : str | torch.device | None, default=None
        Device target for the cloned payload. If ``None``, tensors remain on
        their current device.

    Returns
    -------
    EncodedFeaturePayload
        Cloned feature payload.

    Raises
    ------
    TypeError
        If the payload type is unsupported.

    """
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

    logger.error("Unsupported feature payload type: %s", type(features))
    raise TypeError(f"Unsupported feature payload type: {type(features)!r}")


def _feature_kind(features: EncodedFeaturePayload) -> FeatureKind:
    """Infer the feature payload kind.

    Parameters
    ----------
    features : EncodedFeaturePayload
        Feature payload to inspect.

    Returns
    -------
    FeatureKind
        ``"sam"`` for tensor payloads and ``"sam2"`` for dictionary payloads.

    """
    return "sam2" if isinstance(features, dict) else "sam"


def _normalize_shape(shape: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    """Normalize an image size configuration to ``(height, width)``.

    Parameters
    ----------
    shape : int | tuple[int, int] | list[int]
        Input size description.

    Returns
    -------
    tuple[int, int]
        Normalized image size.

    Raises
    ------
    ValueError
        If the provided shape is invalid.

    """
    if isinstance(shape, int):
        return (shape, shape)
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        return (int(shape[0]), int(shape[1]))

    logger.error("Invalid image shape value: %r", shape)
    raise ValueError(f"Invalid image shape value: {shape!r}")


@dataclass
class EncodedImageFeatures:
    """Encoded image features that can be saved and reused for inference.

    Parameters
    ----------
    checkpoint_path : str
        Checkpoint path used to encode the features.
    src_shape : tuple[int, int]
        Original source image shape as ``(height, width)``.
    dst_shape : tuple[int, int]
        Predictor input shape used when encoding the image.
    features : EncodedFeaturePayload
        Encoded feature payload from a SAM or SAM2 image encoder.

    """

    checkpoint_path: str
    src_shape: Tuple[int, int]
    dst_shape: Tuple[int, int]
    features: EncodedFeaturePayload

    @property
    def feature_kind(self) -> FeatureKind:
        """Return the encoded feature kind.

        Returns
        -------
        FeatureKind
            Feature kind identifier.

        """
        return _feature_kind(self.features)

    def describe(self) -> Dict[str, Any]:
        """Create a compact description of the encoded features.

        Returns
        -------
        dict[str, Any]
            Human-readable feature metadata.

        """
        if isinstance(self.features, torch.Tensor):
            structure: Dict[str, Any] = {"embedding_shape": tuple(self.features.shape)}
        else:
            structure = {
                "image_embed_shape": tuple(self.features["image_embed"].shape),
                "high_res_feat_shapes": [
                    tuple(tensor.shape) for tensor in self.features["high_res_feats"]
                ],
            }

        return {
            "checkpoint_path": self.checkpoint_path,
            "feature_kind": self.feature_kind,
            "src_shape": self.src_shape,
            "dst_shape": self.dst_shape,
            "structure": structure,
        }

    def save(self, file_path: Union[str, Path]) -> Path:
        """Save encoded features to a local file.

        Parameters
        ----------
        file_path : str | Path
            Target feature file path.

        Returns
        -------
        Path
            Saved file path.

        """
        target_path = Path(file_path).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
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
        file_path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
    ) -> EncodedImageFeatures:
        """Load encoded features from a local file.

        Parameters
        ----------
        file_path : str | Path
            Feature file path.
        map_location : str | torch.device, default="cpu"
            Device mapping used by ``torch.load``.

        Returns
        -------
        EncodedImageFeatures
            Loaded encoded features.

        Raises
        ------
        ValueError
            If the loaded file is missing required fields.

        """
        source_path = Path(file_path).expanduser().resolve()
        payload = torch.load(source_path, map_location=map_location)
        required_keys = {"checkpoint_path", "src_shape", "dst_shape", "features"}

        if not isinstance(payload, dict) or not required_keys.issubset(payload):
            logger.error(
                "Feature file has an invalid payload structure: %s", source_path
            )
            raise ValueError(
                f"Feature file has an invalid payload structure: {source_path}"
            )

        return cls(
            checkpoint_path=str(payload["checkpoint_path"]),
            src_shape=tuple(payload["src_shape"]),
            dst_shape=tuple(payload["dst_shape"]),
            features=payload["features"],
        )


class SAMFeatureEncoder:
    """Encode, persist, reload, and reuse Ultralytics SAM image features.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a SAM or SAM2 checkpoint.
    imgsz : int, default=1024
        Predictor image size passed to Ultralytics.
    device : str | torch.device | None, default=None
        Optional inference device.

    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        imgsz: int = 1024,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.imgsz = imgsz
        self.device = device
        self.model = SAM(self.checkpoint_path)

    def _predictor_overrides(self) -> Dict[str, Any]:
        """Build predictor overrides.

        Returns
        -------
        dict[str, Any]
            Predictor overrides passed to the Ultralytics predictor.

        """
        overrides: Dict[str, Any] = {
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "imgsz": self.imgsz,
        }
        if self.device is not None:
            overrides["device"] = str(self.device)
        return overrides

    def _ensure_predictor(self) -> Predictor:
        """Create and configure the underlying Ultralytics predictor.

        Returns
        -------
        Predictor
            Ready-to-use predictor instance.

        """
        if self.model.predictor is None:
            predictor_class = self.model.task_map["segment"]["predictor"]
            predictor = predictor_class(
                overrides=self._predictor_overrides(),
                _callbacks=self.model.callbacks,
            )
            predictor.setup_model(model=self.model.model, verbose=False)
            self.model.predictor = predictor
        return self.model.predictor

    def _validate_checkpoint_match(self, encoded: EncodedImageFeatures) -> None:
        """Ensure that loaded features match the current checkpoint.

        Parameters
        ----------
        encoded : EncodedImageFeatures
            Encoded features to validate.

        Raises
        ------
        ValueError
            If the checkpoint path does not match the current encoder.

        """
        current_name = Path(self.checkpoint_path).name
        encoded_name = Path(encoded.checkpoint_path).name
        if current_name != encoded_name:
            logger.error(
                "Checkpoint mismatch. Current checkpoint is %s but encoded features were produced by %s.",
                current_name,
                encoded_name,
            )
            raise ValueError(
                f"Checkpoint mismatch. Current checkpoint is {current_name!r} "
                f"but encoded features were produced by {encoded_name!r}."
            )

    def _source_shape_from_image(self, image: ImageSource) -> Tuple[int, int]:
        """Extract the original source image shape from the input image.

        Parameters
        ----------
        image : ImageSource
            Image file path or image array.

        Returns
        -------
        tuple[int, int]
            Source shape as ``(height, width)``.

        Raises
        ------
        RuntimeError
            If the source shape cannot be recovered.

        """
        if isinstance(image, np.ndarray):
            if image.ndim < 2:
                logger.error(
                    "NumPy image input must have at least two dimensions, got shape %s.",
                    image.shape,
                )
                raise RuntimeError(
                    "NumPy image input must have at least two dimensions."
                )
            return (int(image.shape[0]), int(image.shape[1]))

        if isinstance(image, (str, Path)):
            image_path = Path(image).expanduser().resolve()
            if not image_path.exists():
                logger.error("Image path does not exist: %s", image_path)
                raise RuntimeError(f"Image path does not exist: {image_path}")

            with Image.open(image_path) as source_image:
                width, height = source_image.size
            return (int(height), int(width))

        logger.error("Unable to determine source shape for image input %r.", image)
        raise RuntimeError(
            f"Unable to determine source shape for image input {image!r}."
        )

    def encode(self, image: ImageSource) -> EncodedImageFeatures:
        """Encode an image into reusable SAM features.

        Parameters
        ----------
        image : ImageSource
            Image file path or image array.

        Returns
        -------
        EncodedImageFeatures
            Encoded image features and metadata.

        Raises
        ------
        RuntimeError
            If feature extraction fails.

        """
        predictor = self._ensure_predictor()
        image_source = str(image) if isinstance(image, Path) else image
        predictor.set_image(image_source)

        if predictor.features is None:
            logger.error("Feature extraction failed for image source %r.", image)
            raise RuntimeError(f"Feature extraction failed for image source {image!r}.")

        src_shape = self._source_shape_from_image(image)
        dst_shape = _normalize_shape(predictor.args.imgsz)
        encoded = EncodedImageFeatures(
            checkpoint_path=self.checkpoint_path,
            src_shape=src_shape,
            dst_shape=dst_shape,
            features=_clone_feature_payload(predictor.features, device="cpu"),
        )
        return encoded

    def encode_to_file(
        self, image: ImageSource, file_path: Union[str, Path]
    ) -> EncodedImageFeatures:
        """Encode an image and save its features to a local file.

        Parameters
        ----------
        image : ImageSource
            Image file path or image array.
        file_path : str | Path
            Target feature file path.

        Returns
        -------
        EncodedImageFeatures
            Encoded feature container.

        """
        encoded = self.encode(image)
        encoded.save(file_path)
        return encoded

    def load_features(
        self,
        file_path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
    ) -> EncodedImageFeatures:
        """Load encoded features from a file and validate the checkpoint.

        Parameters
        ----------
        file_path : str | Path
            Feature file path.
        map_location : str | torch.device, default="cpu"
            Device mapping used by ``torch.load``.

        Returns
        -------
        EncodedImageFeatures
            Loaded and validated features.

        """
        encoded = EncodedImageFeatures.load(file_path, map_location=map_location)
        self._validate_checkpoint_match(encoded)
        return encoded

    @staticmethod
    def _normalize_point_prompts(
        points: PromptPoints | Points | None,
        labels: PromptLabels | None,
    ) -> tuple[PromptPoints | None, PromptLabels | None]:
        """Normalize point prompts for SAM inference.

        Parameters
        ----------
        points : PromptPoints | Points | None
            Raw point coordinates or a :class:`geosam.query.points.Points` object.
        labels : PromptLabels | None
            Explicit point labels.

        Returns
        -------
        tuple[PromptPoints | None, PromptLabels | None]
            Normalized point coordinates and labels.

        Raises
        ------
        ValueError
            If both a labeled :class:`geosam.query.points.Points` object and
            explicit ``labels`` are provided.

        """
        if points is None:
            return None, labels

        from geosam.query.points import Points as QueryPoints

        if isinstance(points, QueryPoints):
            if labels is not None:
                msg = (
                    "Do not provide explicit labels when points is a Points instance. "
                    "Use Points.label instead."
                )
                logger.error(msg)
                raise ValueError(msg)
            return points.to_sam_prompt()
        return points, labels

    def inference_features(
        self,
        encoded: EncodedImageFeatures,
        *,
        bboxes: Optional[PromptBoxes] = None,
        points: PromptPoints | Points | None = None,
        labels: Optional[PromptLabels] = None,
        masks: Optional[PromptMasks] = None,
        multimask_output: bool = False,
        dst_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Run promptable inference using precomputed image features.

        Parameters
        ----------
        encoded : EncodedImageFeatures
            Encoded image features to reuse.
        bboxes : PromptBoxes | None, default=None
            Bounding-box prompts in ``xyxy`` format.
        points : PromptPoints | Points | None, default=None
            Point prompts in pixel coordinates, or a
            :class:`geosam.query.points.Points` object carrying both coordinates
            and labels.
        labels : PromptLabels | None, default=None
            Point labels corresponding to ``points``.
        masks : PromptMasks | None, default=None
            Optional mask prompts.
        multimask_output : bool, default=False
            Whether to request multiple candidate masks.
        dst_shape : tuple[int, int] | None, default=None
            Prompt coordinate space. Defaults to the encoded predictor input size.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor]
            Predicted masks and predicted boxes.

        Raises
        ------
        ValueError
            If point prompts are provided without labels or if the checkpoint
            does not match.

        """
        self._validate_checkpoint_match(encoded)
        points, labels = self._normalize_point_prompts(points, labels)
        if points is not None and labels is None:
            logger.error("Point prompts require matching labels.")
            raise ValueError("Point prompts require matching labels.")

        predictor = self._ensure_predictor()
        predictor_features = _clone_feature_payload(
            encoded.features, device=predictor.device
        )
        return predictor.inference_features(
            features=predictor_features,
            src_shape=encoded.src_shape,
            dst_shape=dst_shape or encoded.dst_shape,
            bboxes=bboxes,
            points=points,
            labels=labels,
            masks=masks,
            multimask_output=multimask_output,
        )

    def inference_feature_file(
        self,
        file_path: Union[str, Path],
        *,
        map_location: Union[str, torch.device] = "cpu",
        bboxes: Optional[PromptBoxes] = None,
        points: PromptPoints | Points | None = None,
        labels: Optional[PromptLabels] = None,
        masks: Optional[PromptMasks] = None,
        multimask_output: bool = False,
        dst_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Load a feature file and run promptable inference from it.

        Parameters
        ----------
        file_path : str | Path
            Feature file path.
        map_location : str | torch.device, default="cpu"
            Device mapping used by ``torch.load``.
        bboxes : PromptBoxes | None, default=None
            Bounding-box prompts in ``xyxy`` format.
        points : PromptPoints | Points | None, default=None
            Point prompts in pixel coordinates, or a
            :class:`geosam.query.points.Points` object carrying both coordinates
            and labels.
        labels : PromptLabels | None, default=None
            Point labels corresponding to ``points``.
        masks : PromptMasks | None, default=None
            Optional mask prompts.
        multimask_output : bool, default=False
            Whether to request multiple candidate masks.
        dst_shape : tuple[int, int] | None, default=None
            Prompt coordinate space. Defaults to the encoded predictor input size.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor]
            Predicted masks and predicted boxes.

        """
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
