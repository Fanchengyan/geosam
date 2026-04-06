"""Online and offline query engines for GeoSAM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import geopandas as gpd
import numpy as np
from pyproj.crs import CRS
from rasterio import Affine
from shapely.geometry import MultiPoint, box

from geosam.datasets import GeoGrid, GridGeoSampler, RasterDataset
from geosam.logging import setup_logger
from geosam.models import EncodedImageFeatures, ModelSpec, build_model_adapter
from geosam.query import (
    BoundingBox,
    Points,
    PromptSet,
    bbox_to_prompt,
    points_to_prompt,
    query_bounds,
    query_center,
    window_from_center,
)

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from geosam.models import GeoSamPrediction
    from geosam.query.prompts import GeoQuery
    from geosam.typing import PathLike

logger = setup_logger(__name__)

MANIFEST_REQUIRED_COLUMNS = {
    "feature_path",
    "chip_id",
    "source_path",
    "checkpoint_path",
    "model_type",
    "transform",
    "shape",
    "crs",
    "dst_shape",
    "chip_center_x",
    "chip_center_y",
    "geometry",
}


@dataclass
class QueryResult:
    """Result returned by online and offline query engines."""

    mask_array: np.ndarray
    mask_transform: Affine
    mask_crs: CRS
    query_bounds: BoundingBox
    chip_bounds: BoundingBox
    scores: np.ndarray
    source_path: str
    chip_id: Optional[str] = None
    model_type: Optional[str] = None


@dataclass
class OnlineQueryCache:
    """Reusable cache for online raster queries."""

    source_path: Optional[str] = None
    chip_bounds: Optional[BoundingBox] = None
    chip_grid: Optional[GeoGrid] = None
    encoded: Optional[EncodedImageFeatures] = None

    def clear(self) -> None:
        """Reset cached online-query state."""
        self.source_path = None
        self.chip_bounds = None
        self.chip_grid = None
        self.encoded = None


def _prediction_to_result(
    prediction: GeoSamPrediction,
    *,
    sample_grid: GeoGrid,
    query_bounds_value: BoundingBox,
    source_path: str,
    chip_id: Optional[str],
    model_type: str,
) -> QueryResult:
    """Convert a model prediction into a public query result."""
    if prediction.masks is None:
        mask_array = np.zeros((0, sample_grid.height, sample_grid.width), dtype=bool)
    else:
        mask_array = prediction.masks.detach().cpu().numpy().astype(bool)
    scores = prediction.scores.detach().cpu().numpy()
    return QueryResult(
        mask_array=mask_array,
        mask_transform=sample_grid.transform,
        mask_crs=sample_grid.crs,
        query_bounds=query_bounds_value,
        chip_bounds=sample_grid.bounds,
        scores=scores,
        source_path=source_path,
        chip_id=chip_id,
        model_type=model_type,
    )


def _require_query_crs(query: GeoQuery) -> None:
    """Require a CRS for public query operations."""
    if query.crs is None:
        msg = "Queries must carry a CRS for GeoSAM operations."
        logger.error(msg)
        raise ValueError(msg)


def _normalize_point_prompt(
    coordinates: Any,
    prompt_labels: Any,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Shape point prompts as a single prompted object with multiple clicks."""
    normalized_coordinates = np.asarray(coordinates, dtype=np.float32)
    normalized_labels = None
    if prompt_labels is not None:
        normalized_labels = np.asarray(prompt_labels, dtype=np.int32)

    if normalized_coordinates.ndim == 2 and normalized_coordinates.shape[1] == 2:
        normalized_coordinates = normalized_coordinates[None, :, :]
        if normalized_labels is not None and normalized_labels.ndim == 1:
            normalized_labels = normalized_labels[None, :]

    return normalized_coordinates, normalized_labels


def _prompt_prediction_kwargs(
    query: GeoQuery,
    chip_grid: GeoGrid,
) -> dict[str, Any]:
    """Build normalized adapter kwargs for a query on a chip grid."""
    if isinstance(query, PromptSet):
        prompt_kwargs: dict[str, Any] = {}
        if query.points is not None:
            point_prompt, point_labels = points_to_prompt(query.points, chip_grid)
            point_prompt, point_labels = _normalize_point_prompt(
                point_prompt,
                point_labels,
            )
            prompt_kwargs["points"] = point_prompt
            prompt_kwargs["labels"] = point_labels
        if query.bbox is not None:
            bbox_prompt = bbox_to_prompt(query.bbox, chip_grid)
            prompt_kwargs["bboxes"] = [list(bbox_prompt)]
        return prompt_kwargs

    if isinstance(query, Points):
        point_prompt, point_labels = points_to_prompt(query, chip_grid)
        point_prompt, point_labels = _normalize_point_prompt(
            point_prompt,
            point_labels,
        )
        return {
            "points": point_prompt,
            "labels": point_labels,
        }

    bbox_prompt = bbox_to_prompt(query, chip_grid)
    return {"bboxes": [list(bbox_prompt)]}


def _query_geometry(query: GeoQuery, target_crs: CRS) -> BaseGeometry:
    """Convert a query to a geometry in the target CRS."""
    if isinstance(query, PromptSet):
        geometries: list[BaseGeometry] = []
        if query.bbox is not None:
            bounds = (
                query.bbox
                if query.bbox.crs == target_crs
                else query.bbox.to_crs(target_crs)
            )
            geometries.append(box(*bounds.to_tuple()))
        if query.points is not None:
            projected_points = (
                query.points
                if query.points.crs == target_crs
                else query.points.to_crs(target_crs)
            )
            if len(projected_points) == 1:
                geometries.append(projected_points.to_geodataframe().geometry.iloc[0])
            else:
                geometries.append(
                    MultiPoint(projected_points.values.tolist()).convex_hull
                )
        if len(geometries) == 1:
            return geometries[0]
        return geometries[0].union(geometries[1])

    if isinstance(query, BoundingBox):
        bounds = query if query.crs == target_crs else query.to_crs(target_crs)
        return box(*bounds.to_tuple())
    projected = query if query.crs == target_crs else query.to_crs(target_crs)
    if len(projected) == 1:
        return projected.to_geodataframe().geometry.iloc[0]
    return MultiPoint(projected.values.tolist())


class OnlineQueryEngine:
    """Execute real-time chip extraction and promptable segmentation."""

    def __init__(self, dataset: RasterDataset, model_spec: ModelSpec) -> None:
        """Initialize an online query engine."""
        self.dataset = dataset
        self.model_spec = model_spec
        self.adapter = build_model_adapter(model_spec)

    def query(
        self,
        query: GeoQuery,
        *,
        multimask_output: bool = False,
        cache: Optional[OnlineQueryCache] = None,
    ) -> QueryResult:
        """Run an online promptable query against the source raster."""
        _require_query_crs(query)
        projected_query = (
            query if query.crs == self.dataset.crs else query.to_crs(self.dataset.crs)
        )

        projected_bounds = query_bounds(projected_query)
        supports_feature_reuse = self.model_spec.resolved_supports_feature_reuse
        should_reencode = True
        if (
            supports_feature_reuse
            and cache is not None
            and cache.chip_bounds is not None
            and cache.chip_grid is not None
            and cache.encoded is not None
            and cache.chip_bounds.contains(projected_bounds)
        ):
            should_reencode = False

        if should_reencode:
            center = query_center(projected_query)
            chip_bounds = window_from_center(
                center,
                self.model_spec.imgsz,
                grid=self.dataset.grid,
            )
            sample = self.dataset[chip_bounds]
            chip_grid = sample.grid
            if cache is None or not supports_feature_reuse:
                prediction = self._predict_from_query(
                    sample=sample,
                    chip_grid=chip_grid,
                    query=query,
                    multimask_output=multimask_output,
                )
                source_path = sample.source_path
                if cache is not None:
                    cache.clear()
            else:
                encoded = self.adapter.encode_image(sample.to_model_image())
                prompt_kwargs = _prompt_prediction_kwargs(query, chip_grid)
                prediction = self.adapter.predict_features(
                    encoded,
                    multimask_output=multimask_output,
                    **prompt_kwargs,
                )
                source_path = sample.source_path
                cache.source_path = source_path
                cache.chip_bounds = chip_bounds
                cache.chip_grid = chip_grid
                cache.encoded = encoded
        else:
            chip_bounds = cache.chip_bounds
            chip_grid = cache.chip_grid
            prompt_kwargs = _prompt_prediction_kwargs(query, chip_grid)
            prediction = self.adapter.predict_features(
                cache.encoded,
                multimask_output=multimask_output,
                **prompt_kwargs,
            )
            source_path = cache.source_path or str(self.dataset.source)

        return _prediction_to_result(
            prediction,
            sample_grid=chip_grid,
            query_bounds_value=projected_bounds,
            source_path=source_path,
            chip_id=None,
            model_type=self.model_spec.model_type,
        )

    def _predict_from_query(
        self,
        *,
        sample: Any,
        chip_grid: GeoGrid,
        query: GeoQuery,
        multimask_output: bool,
    ) -> GeoSamPrediction:
        """Convert a query to prompts and run prediction."""
        prompt_kwargs = _prompt_prediction_kwargs(query, chip_grid)
        return self.adapter.predict_image(
            sample.to_model_image(),
            multimask_output=multimask_output,
            **prompt_kwargs,
        )


class FeatureCacheBuilder:
    """Build a local feature cache and GeoParquet manifest."""

    def __init__(
        self,
        dataset: RasterDataset,
        model_spec: ModelSpec,
        output_dir: PathLike,
        *,
        chip_size: Optional[Union[int, tuple[int, int]]] = None,
        stride: Optional[Union[int, tuple[int, int]]] = None,
        overlap: Optional[Union[int, tuple[int, int]]] = None,
    ) -> None:
        """Initialize a feature cache builder."""
        self.dataset = dataset
        self.model_spec = model_spec
        self.adapter = build_model_adapter(model_spec)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir = self.output_dir / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.chip_size = chip_size or model_spec.imgsz
        self.stride = stride
        self.overlap = overlap

    def build(self, manifest_path: Optional[PathLike] = None) -> Path:
        """Encode all chips and write a manifest."""
        if not self.model_spec.resolved_supports_feature_reuse:
            msg = "Feature cache building requires a model with feature-reuse support."
            logger.error(msg)
            raise NotImplementedError(msg)

        manifest_target = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else self.output_dir / "manifest.parquet"
        )

        sampler = GridGeoSampler(
            self.dataset,
            chip_size=self.chip_size,
            stride=self.stride,
            overlap=self.overlap,
        )
        rows: list[dict[str, Any]] = []
        for index, chip_bounds in enumerate(sampler):
            sample = self.dataset[chip_bounds]
            encoded = self.adapter.encode_image(sample.to_model_image())
            chip_id = f"chip_{index:06d}"
            feature_path = self.features_dir / f"{chip_id}.pt"
            encoded.save(feature_path)
            rows.append(
                {
                    "feature_path": str(feature_path),
                    "chip_id": chip_id,
                    "source_path": sample.source_path,
                    "checkpoint_path": encoded.checkpoint_path,
                    "model_type": encoded.model_type,
                    "transform": json.dumps(list(sample.transform)[:6]),
                    "shape": json.dumps(list(sample.shape)),
                    "crs": sample.crs.to_string(),
                    "dst_shape": json.dumps(list(encoded.dst_shape)),
                    "chip_center_x": sample.bbox.center[0],
                    "chip_center_y": sample.bbox.center[1],
                    "geometry": sample.bbox.to_geometry(),
                }
            )

        manifest = gpd.GeoDataFrame(rows, geometry="geometry", crs=self.dataset.crs)
        return self.write_manifest(manifest, manifest_target)

    def write_manifest(
        self,
        manifest: gpd.GeoDataFrame,
        manifest_path: Path,
    ) -> Path:
        """Write the feature manifest as GeoParquet."""
        try:
            manifest.to_parquet(manifest_path)
        except ModuleNotFoundError as exc:
            msg = (
                "Writing a GeoParquet manifest requires pyarrow. "
                "Install pyarrow to enable feature cache manifests."
            )
            logger.exception(msg)
            raise ModuleNotFoundError(msg) from exc
        return manifest_path


class FeatureQueryEngine:
    """Run promptable queries against a cached feature manifest."""

    def __init__(self, manifest_path: PathLike, model_spec: ModelSpec) -> None:
        """Initialize a feature-query engine."""
        self.manifest_path = self.resolve_manifest_path(manifest_path)
        self.model_spec = model_spec
        self.adapter = build_model_adapter(model_spec)
        self.manifest = self.load_manifest(self.manifest_path)
        missing_columns = MANIFEST_REQUIRED_COLUMNS.difference(self.manifest.columns)
        if missing_columns:
            msg = f"Manifest is missing required columns: {sorted(missing_columns)}"
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def load_manifest(manifest_path: Path) -> gpd.GeoDataFrame:
        """Load a GeoParquet feature manifest."""
        try:
            return gpd.read_parquet(manifest_path)
        except ModuleNotFoundError as exc:
            msg = (
                "Reading a GeoParquet manifest requires pyarrow. "
                "Install pyarrow to query cached features."
            )
            logger.exception(msg)
            raise ModuleNotFoundError(msg) from exc

    @staticmethod
    def resolve_manifest_path(manifest_path: PathLike) -> Path:
        """Resolve a manifest file path from either a file or a feature folder."""
        candidate = Path(manifest_path).expanduser().resolve()
        if candidate.is_file():
            return candidate
        if not candidate.is_dir():
            msg = f"Manifest path does not exist: {candidate}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        preferred_paths = [
            candidate / "manifest.parquet",
            candidate / "manifest.pkl",
        ]
        for preferred_path in preferred_paths:
            if preferred_path.exists():
                return preferred_path

        manifest_matches = sorted(candidate.glob("*.parquet")) + sorted(
            candidate.glob("*.pkl")
        )
        if len(manifest_matches) == 1:
            return manifest_matches[0]
        if len(manifest_matches) == 0:
            msg = f"No manifest file found in feature folder: {candidate}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        msg = (
            "Multiple manifest candidates were found in the feature folder. "
            "Please choose one explicitly: "
            f"{[str(path.name) for path in manifest_matches]}"
        )
        logger.error(msg)
        raise ValueError(msg)

    def query(
        self,
        query: GeoQuery,
        *,
        multimask_output: bool = False,
    ) -> QueryResult:
        """Run a promptable query against the cached feature manifest."""
        _require_query_crs(query)
        target_crs = CRS.from_user_input(self.manifest.crs)
        projected_query = query if query.crs == target_crs else query.to_crs(target_crs)

        projected_bounds = query_bounds(projected_query)

        query_geometry = _query_geometry(projected_query, target_crs)
        candidates = self.manifest[
            self.manifest.geometry.apply(
                lambda geometry: geometry.covers(query_geometry)
            )
        ]
        if len(candidates) == 0:
            msg = "No cached chip fully covers the requested query."
            logger.error(msg)
            raise ValueError(msg)

        center = query_center(projected_query)
        best_index = self._nearest_candidate_index(candidates, center=center)
        best_row = candidates.iloc[best_index]
        chip_grid = self._chip_grid_from_row(best_row)
        encoded = EncodedImageFeatures.load(best_row["feature_path"])

        if isinstance(projected_query, (PromptSet, Points)):
            prompt_kwargs = _prompt_prediction_kwargs(projected_query, chip_grid)
            prediction = self.adapter.predict_features(
                encoded,
                multimask_output=multimask_output,
                **prompt_kwargs,
            )
        else:
            prompt_kwargs = _prompt_prediction_kwargs(projected_query, chip_grid)
            prediction = self.adapter.predict_features(
                encoded,
                multimask_output=multimask_output,
                **prompt_kwargs,
            )

        return _prediction_to_result(
            prediction,
            sample_grid=chip_grid,
            query_bounds_value=projected_bounds,
            source_path=str(best_row["source_path"]),
            chip_id=str(best_row["chip_id"]),
            model_type=str(best_row["model_type"]),
        )

    @staticmethod
    def _nearest_candidate_index(
        candidates: gpd.GeoDataFrame,
        *,
        center: tuple[float, float],
    ) -> int:
        """Return the index position of the nearest chip center."""
        distances = (
            candidates["chip_center_x"].to_numpy(dtype=float) - center[0]
        ) ** 2 + (candidates["chip_center_y"].to_numpy(dtype=float) - center[1]) ** 2
        return int(np.argmin(distances))

    @staticmethod
    def _chip_grid_from_row(row: Any) -> GeoGrid:
        """Reconstruct a chip GeoGrid from a manifest row."""
        transform_values = json.loads(row["transform"])
        shape = tuple(json.loads(row["shape"]))
        crs = CRS.from_user_input(row["crs"])
        return GeoGrid(Affine(*transform_values), shape, crs)
