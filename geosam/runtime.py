"""Shared runtime helpers for GeoSAM model/query workflows.

This module provides package-level utilities that are independent from any
particular host application. It covers model metadata, checkpoint inference,
feature-source inspection, dependency discovery, and raster chip extent
enumeration.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

from geosam.crs import crs_equal
from geosam.datasets import RasterDataset
from geosam.logging import setup_logger
from geosam.models import ModelSpec
from geosam.query import BoundingBox
from geosam.query.prompts import normalize_chip_size

logger = setup_logger(__name__)

DEFAULT_MODEL_REPOSITORY = "https://github.com/Fanchengyan/geosam-models"


@dataclass(frozen=True)
class ModelDefinition:
    """Supported downloadable model metadata.

    Parameters
    ----------
    model_id : str
        Stable identifier exposed to clients.
    label : str
        Human-readable model label.
    model_type : {"sam", "sam2", "sam3"}
        Underlying model family.
    filename : str
        Expected checkpoint filename.
    supports_feature_reuse : bool, optional
        Whether the model supports reusable encoded features.

    """

    model_id: str
    label: str
    model_type: Literal["sam", "sam2", "sam3"]
    filename: str
    supports_feature_reuse: bool = True


@dataclass(frozen=True)
class FeatureSourceSummary:
    """Summary metadata for a cached GeoSAM feature source.

    Parameters
    ----------
    manifest_path : Path
        Resolved manifest file path.
    crs_text : str
        Stable CRS string for downstream consumers.
    extent : tuple[float, float, float, float]
        Total bounds in ``(min_x, min_y, max_x, max_y)`` order.
    chip_count : int
        Number of cached chips in the manifest.
    pixel_area : float
        Approximate area represented by one source pixel.

    """

    manifest_path: Path
    crs_text: str
    extent: tuple[float, float, float, float]
    chip_count: int
    pixel_area: float


MODEL_DEFINITIONS: tuple[ModelDefinition, ...] = (
    ModelDefinition("sam_b", "SAM Base", "sam", "sam_b.pt"),
    ModelDefinition("sam_l", "SAM Large", "sam", "sam_l.pt"),
    ModelDefinition("sam2_t", "SAM2 Tiny", "sam2", "sam2_t.pt"),
    ModelDefinition("sam2_s", "SAM2 Small", "sam2", "sam2_s.pt"),
    ModelDefinition("sam2_b", "SAM2 Base", "sam2", "sam2_b.pt"),
    ModelDefinition("sam2_l", "SAM2 Large", "sam2", "sam2_l.pt"),
    ModelDefinition("sam2.1_t", "SAM2.1 Tiny", "sam2", "sam2.1_t.pt"),
    ModelDefinition("sam2.1_s", "SAM2.1 Small", "sam2", "sam2.1_s.pt"),
    ModelDefinition("sam2.1_b", "SAM2.1 Base", "sam2", "sam2.1_b.pt"),
    ModelDefinition("sam2.1_l", "SAM2.1 Large", "sam2", "sam2.1_l.pt"),
    ModelDefinition("sam3", "SAM3", "sam3", "sam3.pt", supports_feature_reuse=True),
    ModelDefinition(
        "sam3.1_multiplex",
        "SAM3.1 Multiplex",
        "sam3",
        "sam3.1_multiplex.pt",
        supports_feature_reuse=True,
    ),
)


def get_model_definition(model_id: str) -> ModelDefinition:
    """Return model metadata for a registered model id.

    Parameters
    ----------
    model_id : str
        Target model identifier.

    Returns
    -------
    ModelDefinition
        Matching model definition.

    Raises
    ------
    KeyError
        If ``model_id`` is not registered.

    """
    for definition in MODEL_DEFINITIONS:
        if definition.model_id == model_id:
            return definition

    logger.error("Unknown GeoSAM model id requested: %s", model_id)
    raise KeyError(model_id)


def get_model_display_items() -> list[tuple[str, str]]:
    """Return selectable model entries.

    Returns
    -------
    list[tuple[str, str]]
        Pairs of ``(model_id, label)``.

    """
    return [(definition.model_id, definition.label) for definition in MODEL_DEFINITIONS]


def infer_model_id_from_checkpoint_path(
    checkpoint_path: Union[str, Path],
    *,
    fallback_model_id: Optional[str] = None,
) -> str:
    """Infer a registered model id from a checkpoint path.

    Parameters
    ----------
    checkpoint_path : str | Path
        Checkpoint file path.
    fallback_model_id : str | None, optional
        Fallback model id returned when filename inference fails.

    Returns
    -------
    str
        Resolved model id.

    Raises
    ------
    ValueError
        If the path cannot be mapped to a known model and no fallback is given.

    """
    checkpoint_name = Path(checkpoint_path).name.lower()
    for definition in MODEL_DEFINITIONS:
        if checkpoint_name == definition.filename.lower():
            return definition.model_id
    for definition in MODEL_DEFINITIONS:
        stem = Path(definition.filename).stem.lower()
        if stem in checkpoint_name:
            return definition.model_id
    if fallback_model_id is not None:
        return fallback_model_id

    msg = (
        "Could not infer a GeoSAM model id from the checkpoint path. "
        "Please select a supported model explicitly."
    )
    logger.error(msg)
    raise ValueError(msg)


def create_model_spec(
    model_id: str,
    checkpoint_path: Union[str, Path],
    *,
    device: Optional[str] = None,
) -> ModelSpec:
    """Create a model spec for a registered model and explicit checkpoint path.

    Parameters
    ----------
    model_id : str
        Registered GeoSAM model identifier.
    checkpoint_path : str | Path
        Model checkpoint path.
    device : str | None, optional
        Optional inference device string.

    Returns
    -------
    ModelSpec
        Model specification compatible with GeoSAM adapters.

    """
    definition = get_model_definition(model_id)
    return ModelSpec(
        model_type=definition.model_type,
        checkpoint_path=checkpoint_path,
        device=device,
        supports_feature_reuse=definition.supports_feature_reuse,
    )


def create_model_spec_from_checkpoint(
    checkpoint_path: Union[str, Path],
    *,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> ModelSpec:
    """Create a model spec from an arbitrary checkpoint path.

    Parameters
    ----------
    checkpoint_path : str | Path
        Checkpoint file path.
    model_id : str | None, optional
        Optional fallback model id.
    device : str | None, optional
        Optional inference device string.

    Returns
    -------
    ModelSpec
        Model specification compatible with GeoSAM adapters.

    """
    resolved_model_id = infer_model_id_from_checkpoint_path(
        checkpoint_path,
        fallback_model_id=model_id,
    )
    return create_model_spec(
        resolved_model_id,
        checkpoint_path,
        device=device,
    )


def dependency_status() -> dict[str, bool]:
    """Return import availability for GeoSAM runtime dependencies.

    Returns
    -------
    dict[str, bool]
        Mapping of module name to availability flag.

    """
    module_names = [
        "geosam",
        "torch",
        "ultralytics",
        "rasterio",
        "geopandas",
        "pyarrow",
    ]
    return {
        module_name: importlib.util.find_spec(module_name) is not None
        for module_name in module_names
    }


def resolve_feature_manifest_path(feature_dir: Union[str, Path]) -> Path:
    """Resolve the manifest path for a feature folder.

    Parameters
    ----------
    feature_dir : str | Path
        Feature directory or manifest path.

    Returns
    -------
    Path
        Resolved manifest path.

    """
    from geosam.engines import FeatureQueryEngine

    return FeatureQueryEngine.resolve_manifest_path(feature_dir)


def describe_feature_source(feature_dir: Union[str, Path]) -> FeatureSourceSummary:
    """Load summary metadata for a GeoSAM feature folder.

    Parameters
    ----------
    feature_dir : str | Path
        Feature directory or manifest path.

    Returns
    -------
    FeatureSourceSummary
        Summary information for the feature source.

    """
    import geopandas as gpd
    import pandas as pd

    manifest_path = resolve_feature_manifest_path(feature_dir)
    if manifest_path.suffix == ".pkl":
        frame = pd.read_pickle(manifest_path)
        frame = frame.set_crs(frame.crs or "EPSG:4326", allow_override=True)
    else:
        frame = gpd.read_parquet(manifest_path)

    crs_text = _resolve_feature_crs_text(frame)
    total_bounds = tuple(float(value) for value in frame.total_bounds)
    transform_values = json.loads(frame.iloc[0]["transform"])
    pixel_area = abs(float(transform_values[0]) * float(transform_values[4]))
    return FeatureSourceSummary(
        manifest_path=manifest_path,
        crs_text=crs_text,
        extent=total_bounds,
        chip_count=len(frame),
        pixel_area=pixel_area,
    )


def _resolve_feature_crs_text(frame: Any) -> str:
    """Resolve a stable CRS string from a manifest frame.

    Parameters
    ----------
    frame : Any
        GeoPandas or Pandas frame containing feature metadata.

    Returns
    -------
    str
        Stable CRS string.

    Raises
    ------
    ValueError
        If CRS metadata is missing.

    """
    if "crs" in frame.columns and len(frame) > 0:
        manifest_crs = frame.iloc[0]["crs"]
        if isinstance(manifest_crs, str) and manifest_crs.strip():
            return manifest_crs.strip()

    frame_crs = frame.crs
    if frame_crs is None:
        msg = "Feature manifest CRS is missing."
        logger.error(msg)
        raise ValueError(msg)

    to_authority = getattr(frame_crs, "to_authority", None)
    if callable(to_authority):
        authority = to_authority()
        if authority is not None:
            return f"{authority[0]}:{authority[1]}"

    to_epsg = getattr(frame_crs, "to_epsg", None)
    if callable(to_epsg):
        epsg_code = to_epsg()
        if epsg_code is not None:
            return f"EPSG:{epsg_code}"

    to_wkt = getattr(frame_crs, "to_wkt", None)
    if callable(to_wkt):
        return str(to_wkt())

    return str(frame_crs)


def chip_extent_rectangles_for_source(
    source_path: Union[str, Path],
    *,
    bands: Optional[list[int]] = None,
    crs: Optional[str] = None,
    res: Optional[float] = None,
    extent: Optional[tuple[float, float, float, float]] = None,
    extent_crs: Optional[str] = None,
    chip_size: int = 1024,
    stride: int = 512,
) -> list[tuple[float, float, float, float]]:
    """Return chip extents for a raster source using GeoSAM sampling rules.

    Parameters
    ----------
    source_path : str | Path
        Raster source path.
    bands : list[int] | None, optional
        Raster band indexes.
    crs : str | None, optional
        Source CRS override.
    res : float | None, optional
        Source resolution override.
    extent : tuple[float, float, float, float] | None, optional
        ROI bounds in ``(min_x, min_y, max_x, max_y)`` order.
    extent_crs : str | None, optional
        CRS string for ``extent``.
    chip_size : int, optional
        Chip size in pixels.
    stride : int, optional
        Sliding stride in pixels.

    Returns
    -------
    list[tuple[float, float, float, float]]
        Chip bounds in ``(left, bottom, right, top)`` order.

    """
    dataset = RasterDataset(
        source_path,
        indexes=bands,
        crs=crs,
        res=res,
    )
    roi_bounds = dataset.bounds
    if extent is not None:
        extent_bounds = BoundingBox(
            extent[0],
            extent[1],
            extent[2],
            extent[3],
            crs=extent_crs or dataset.crs,
        )
        roi_bounds = (
            extent_bounds
            if crs_equal(extent_bounds.crs, dataset.crs)
            else extent_bounds.to_crs(dataset.crs)
        )
        intersection = roi_bounds & dataset.bounds
        if intersection is None:
            return []
        roi_bounds = intersection

    roi_grid = dataset.grid.to_view(roi_bounds)
    chip_height, chip_width = normalize_chip_size(chip_size)
    stride_height, stride_width = normalize_chip_size(stride)

    row_starts = _window_starts(roi_grid.height, chip_height, stride_height)
    col_starts = _window_starts(roi_grid.width, chip_width, stride_width)

    rectangles: list[tuple[float, float, float, float]] = []
    for row_start in row_starts:
        for col_start in col_starts:
            chip_grid = roi_grid.window(
                row_off=row_start,
                col_off=col_start,
                height=min(chip_height, roi_grid.height),
                width=min(chip_width, roi_grid.width),
            )
            bounds = chip_grid.bounds
            rectangles.append((bounds.left, bounds.bottom, bounds.right, bounds.top))
    return rectangles


def _window_starts(size: int, tile_size: int, tile_stride: int) -> list[int]:
    """Return sliding-window start positions.

    Parameters
    ----------
    size : int
        Total axis size.
    tile_size : int
        Window size.
    tile_stride : int
        Window stride.

    Returns
    -------
    list[int]
        Window start indices.

    """
    if tile_size >= size:
        return [0]

    starts = list(range(0, max(size - tile_size, 0) + 1, tile_stride))
    last_start = size - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts
