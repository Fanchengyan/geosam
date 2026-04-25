"""Mask vectorization and GeoJSON export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from shapely.geometry import mapping
from shapely.geometry import shape as shapely_shape

from geosam.logging import setup_logger

if TYPE_CHECKING:
    from rasterio import Affine
    from shapely.geometry.base import BaseGeometry

    from geosam.engines import QueryResult
    from geosam.typing import CrsLike, PathLike

logger = setup_logger(__name__)


class MaskVectorizer:
    """Polygonize raster masks and export them as vector data."""

    def __init__(
        self,
        mask_array: np.ndarray,
        *,
        transform: Affine,
        crs: CrsLike,
        properties: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a bound mask vectorizer.

        Parameters
        ----------
        mask_array : np.ndarray
            Mask array bound to this vectorizer instance.
        transform : Affine
            Affine transform associated with ``mask_array``.
        crs : CrsLike
            Coordinate reference system associated with ``mask_array``.
        properties : dict[str, Any] | None, optional
            Default feature properties attached during export.

        """
        self.mask_array = mask_array
        self.transform = transform
        self.crs = crs
        self.properties = {} if properties is None else dict(properties)

    @classmethod
    def from_query_result(
        cls,
        query_result: QueryResult,
        *,
        properties: Optional[dict[str, Any]] = None,
    ) -> MaskVectorizer:
        """Build a vectorizer bound to a :class:`geosam.engines.QueryResult`.

        Parameters
        ----------
        query_result : QueryResult
            Query result carrying masks and geospatial metadata.
        properties : dict[str, Any] | None, optional
            Additional default properties to attach during export.

        Returns
        -------
        MaskVectorizer
            Bound vectorizer instance.

        """
        base_properties = {
            "source_path": query_result.source_path,
            "model_type": query_result.model_type,
            "chip_id": query_result.chip_id,
        }
        if properties is not None:
            base_properties.update(properties)
        return cls(
            query_result.mask_array,
            transform=query_result.mask_transform,
            crs=query_result.mask_crs,
            properties=base_properties,
        )

    def to_geodataframe(
        self,
        *,
        mask_index: int = 0,
        properties: Optional[dict[str, Any]] = None,
        simplify_tolerance: float | None = None,
        preserve_topology: bool = True,
    ) -> gpd.GeoDataFrame:
        """Convert the bound mask array into polygon features.

        Parameters
        ----------
        mask_index : int, optional
            Index of the mask to export when multiple masks are available.
        properties : dict[str, Any] | None, optional
            Additional properties to merge with the bound default properties.
        simplify_tolerance : float | None, optional
            Optional tolerance passed to Shapely ``simplify`` before export.
        preserve_topology : bool, default=True
            Whether topology should be preserved during simplification.

        Returns
        -------
        gpd.GeoDataFrame
            Polygonized mask features.

        """
        merged_properties = dict(self.properties)
        if properties is not None:
            merged_properties.update(properties)
        return _vectorize_mask(
            self.mask_array,
            transform=self.transform,
            crs=self.crs,
            mask_index=mask_index,
            properties=merged_properties,
            simplify_tolerance=simplify_tolerance,
            preserve_topology=preserve_topology,
        )

    def to_geometries(
        self,
        *,
        mask_index: int = 0,
        simplify_tolerance: float | None = None,
        preserve_topology: bool = True,
    ) -> list[BaseGeometry]:
        """Convert the bound mask array into Shapely geometries.

        Parameters
        ----------
        mask_index : int, optional
            Index of the mask to export when multiple masks are available.
        simplify_tolerance : float | None, optional
            Optional tolerance passed to Shapely ``simplify`` before export.
        preserve_topology : bool, default=True
            Whether topology should be preserved during simplification.

        Returns
        -------
        list[BaseGeometry]
            Polygonized mask geometries.

        """
        return _vectorize_mask_geometries(
            self.mask_array,
            transform=self.transform,
            mask_index=mask_index,
            simplify_tolerance=simplify_tolerance,
            preserve_topology=preserve_topology,
        )

    def to_preview_geometries(
        self,
        *,
        mask_index: int = 0,
        simplify_tolerance: float | None = None,
        preserve_topology: bool = True,
    ) -> list[BaseGeometry]:
        """Return geometries optimized for interactive preview rendering."""
        return self.to_geometries(
            mask_index=mask_index,
            simplify_tolerance=simplify_tolerance,
            preserve_topology=preserve_topology,
        )

    def to_geojson(
        self,
        *,
        mask_index: int = 0,
        properties: Optional[dict[str, Any]] = None,
        simplify_tolerance: float | None = None,
        preserve_topology: bool = True,
    ) -> dict[str, Any]:
        """Convert the bound mask result to a GeoJSON dictionary."""
        merged_properties = dict(self.properties)
        if properties is not None:
            merged_properties.update(properties)
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": dict(merged_properties),
                    "geometry": mapping(geometry),
                }
                for geometry in self.to_geometries(
                    mask_index=mask_index,
                    simplify_tolerance=simplify_tolerance,
                    preserve_topology=preserve_topology,
                )
            ],
        }

    def write_geojson(
        self,
        output_path: PathLike,
        *,
        mask_index: int = 0,
        properties: Optional[dict[str, Any]] = None,
        simplify_tolerance: float | None = None,
        preserve_topology: bool = True,
    ) -> Path:
        """Write the bound mask result to a GeoJSON file."""
        payload = self.to_geojson(
            mask_index=mask_index,
            properties=properties,
            simplify_tolerance=simplify_tolerance,
            preserve_topology=preserve_topology,
        )
        target_path = Path(output_path).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return target_path


def _select_mask(mask_array: np.ndarray, mask_index: int = 0) -> np.ndarray:
    """Select a single 2D mask from a mask stack.

    Parameters
    ----------
    mask_array : np.ndarray
        Input mask array with shape ``(height, width)`` or
        ``(num_masks, height, width)``.
    mask_index : int, optional
        Index of the mask to select when ``mask_array`` contains multiple masks.

    Returns
    -------
    np.ndarray
        Selected 2D boolean mask.

    Raises
    ------
    ValueError
        If ``mask_array`` does not have 2 or 3 dimensions.
    IndexError
        If ``mask_index`` is out of range.

    """
    if mask_array.ndim == 2:
        return mask_array.astype(bool)
    if mask_array.ndim != 3:
        msg = f"Mask array must have 2 or 3 dimensions. Got {mask_array.shape}."
        logger.error(msg)
        raise ValueError(msg)
    if not 0 <= mask_index < mask_array.shape[0]:
        msg = (
            f"mask_index {mask_index} is out of range for {mask_array.shape[0]} masks."
        )
        logger.error(msg)
        raise IndexError(msg)
    return mask_array[mask_index].astype(bool)


def _vectorize_mask(
    mask_array: np.ndarray,
    *,
    transform: Affine,
    crs: CrsLike,
    mask_index: int = 0,
    properties: Optional[dict[str, Any]] = None,
    simplify_tolerance: float | None = None,
    preserve_topology: bool = True,
) -> gpd.GeoDataFrame:
    """Convert a mask array into polygon features.

    Parameters
    ----------
    mask_array : np.ndarray
        Input mask array with shape ``(height, width)`` or
        ``(num_masks, height, width)``.
    transform : Affine
        Affine transform of the mask array.
    crs : CrsLike
        Coordinate reference system of the mask array.
    mask_index : int, optional
        Index of the mask to export when multiple masks are available.
    properties : dict[str, Any] | None, optional
        Properties attached to each exported feature.
    simplify_tolerance : float | None, optional
        Optional tolerance passed to Shapely ``simplify`` before export.
    preserve_topology : bool, default=True
        Whether topology should be preserved during simplification.

    Returns
    -------
    gpd.GeoDataFrame
        Polygonized mask features.

    """
    geometries = _vectorize_mask_geometries(
        mask_array,
        transform=transform,
        mask_index=mask_index,
        simplify_tolerance=simplify_tolerance,
        preserve_topology=preserve_topology,
    )
    base_properties = properties or {}
    records = [{**base_properties, "geometry": geometry} for geometry in geometries]

    if len(records) == 0:
        return gpd.GeoDataFrame(
            columns=[*base_properties.keys(), "geometry"],
            geometry="geometry",
            crs=crs,
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def _vectorize_mask_geometries(
    mask_array: np.ndarray,
    *,
    transform: Affine,
    mask_index: int = 0,
    simplify_tolerance: float | None = None,
    preserve_topology: bool = True,
) -> list[BaseGeometry]:
    """Convert a mask array into Shapely geometries."""
    mask = _select_mask(mask_array, mask_index=mask_index)
    geometries: list[BaseGeometry] = []
    for geometry, value in shapes(
        mask.astype(np.uint8),
        mask=mask,
        transform=transform,
    ):
        if int(value) != 1:
            continue
        polygon = shapely_shape(geometry)
        if simplify_tolerance is not None and simplify_tolerance > 0:
            polygon = polygon.simplify(
                simplify_tolerance,
                preserve_topology=preserve_topology,
            )
        if polygon.is_empty:
            continue
        geometries.append(polygon)
    return geometries
