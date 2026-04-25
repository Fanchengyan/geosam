"""Coordinate reference system backends for :mod:`geosam`.

The native backend uses :mod:`pyproj`. The QGIS backend imports QGIS bindings
only when selected, allowing QGIS plugins to avoid importing ``pyproj`` inside
QGIS background tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from geosam.context import get_runtime

if TYPE_CHECKING:
    from geosam.query.bbox import BoundingBox


class CrsBackend(Protocol):
    """CRS operation interface implemented by runtime backends."""

    def normalize_crs(self, crs: Any) -> Any:
        """Normalize user CRS input into a backend-specific CRS object."""

    def crs_equal(self, left: Any, right: Any) -> bool:
        """Return whether two CRS values describe the same CRS."""

    def crs_to_string(self, crs: Any) -> str:
        """Return a stable CRS string."""

    def transform_points(
        self,
        x_values: Any,
        y_values: Any,
        source_crs: Any,
        target_crs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform x/y coordinates between coordinate reference systems."""

    def transform_bounds(
        self,
        bounds: BoundingBox,
        target_crs: Any,
    ) -> tuple[float, float, float, float]:
        """Transform bounding-box coordinates into a target CRS."""


class NativeCrsBackend:
    """CRS backend implemented with :mod:`pyproj` and :mod:`rasterio`."""

    def normalize_crs(self, crs: Any) -> Any:
        """Normalize CRS input with :class:`pyproj.crs.CRS`."""
        from pyproj.crs import CRS

        return CRS.from_user_input(crs)

    def crs_equal(self, left: Any, right: Any) -> bool:
        """Return whether two CRS values are equal in native CRS semantics."""
        return self.normalize_crs(left) == self.normalize_crs(right)

    def crs_to_string(self, crs: Any) -> str:
        """Return a native CRS string."""
        normalized_crs = self.normalize_crs(crs)
        return normalized_crs.to_string()

    def transform_points(
        self,
        x_values: Any,
        y_values: Any,
        source_crs: Any,
        target_crs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform points with :class:`pyproj.Transformer`."""
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            self.normalize_crs(source_crs),
            self.normalize_crs(target_crs),
            always_xy=True,
        )
        transformed_x, transformed_y = transformer.transform(x_values, y_values)
        return np.asarray(transformed_x), np.asarray(transformed_y)

    def transform_bounds(
        self,
        bounds: BoundingBox,
        target_crs: Any,
    ) -> tuple[float, float, float, float]:
        """Transform bounds with :func:`rasterio.warp.transform_bounds`."""
        from rasterio.warp import transform_bounds

        return transform_bounds(
            self.normalize_crs(bounds.crs),
            self.normalize_crs(target_crs),
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )


class QgisCrsBackend:
    """CRS backend implemented with QGIS core classes."""

    def normalize_crs(self, crs: Any) -> Any:
        """Normalize CRS input into ``QgsCoordinateReferenceSystem``."""
        from qgis.core import QgsCoordinateReferenceSystem

        if isinstance(crs, QgsCoordinateReferenceSystem):
            return crs
        if hasattr(crs, "authid"):
            return QgsCoordinateReferenceSystem(crs.authid())
        if hasattr(crs, "to_string"):
            crs = crs.to_string()
        elif hasattr(crs, "to_epsg") and crs.to_epsg() is not None:
            crs = f"EPSG:{crs.to_epsg()}"
        qgis_crs = QgsCoordinateReferenceSystem()
        if isinstance(crs, int):
            qgis_crs.createFromEpsg(crs)
        elif isinstance(crs, str):
            qgis_crs.createFromUserInput(crs)
        else:
            qgis_crs.createFromUserInput(str(crs))
        return qgis_crs

    def crs_equal(self, left: Any, right: Any) -> bool:
        """Return whether two QGIS CRS values are equal."""
        return self.normalize_crs(left) == self.normalize_crs(right)

    def crs_to_string(self, crs: Any) -> str:
        """Return an auth id or WKT for a QGIS CRS."""
        qgis_crs = self.normalize_crs(crs)
        auth_id = qgis_crs.authid()
        if auth_id:
            return str(auth_id)
        return str(qgis_crs.toWkt())

    def _transform_context(self) -> Any:
        """Return the configured QGIS coordinate transform context."""
        from qgis.core import QgsProject

        runtime = get_runtime()
        if runtime.qgis_transform_context is not None:
            return runtime.qgis_transform_context
        if runtime.qgis_project is not None:
            return runtime.qgis_project.transformContext()
        return QgsProject.instance().transformContext()

    def transform_points(
        self,
        x_values: Any,
        y_values: Any,
        source_crs: Any,
        target_crs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform points with ``QgsCoordinateTransform``."""
        from qgis.core import QgsCoordinateTransform, QgsPointXY

        transformer = QgsCoordinateTransform(
            self.normalize_crs(source_crs),
            self.normalize_crs(target_crs),
            self._transform_context(),
        )
        x_array = np.asarray(x_values)
        y_array = np.asarray(y_values)
        transformed_x = np.empty(x_array.shape, dtype=np.float64)
        transformed_y = np.empty(y_array.shape, dtype=np.float64)
        for index, (x_value, y_value) in enumerate(zip(x_array.flat, y_array.flat)):
            point = transformer.transform(QgsPointXY(float(x_value), float(y_value)))
            transformed_x.flat[index] = point.x()
            transformed_y.flat[index] = point.y()
        return (
            transformed_x.reshape(x_array.shape),
            transformed_y.reshape(y_array.shape),
        )

    def transform_bounds(
        self,
        bounds: BoundingBox,
        target_crs: Any,
    ) -> tuple[float, float, float, float]:
        """Transform bounds with ``QgsCoordinateTransform``."""
        from qgis.core import QgsCoordinateTransform, QgsRectangle

        transformer = QgsCoordinateTransform(
            self.normalize_crs(bounds.crs),
            self.normalize_crs(target_crs),
            self._transform_context(),
        )
        rectangle = QgsRectangle(bounds.left, bounds.bottom, bounds.right, bounds.top)
        transformed = transformer.transformBoundingBox(rectangle)
        return (
            float(transformed.xMinimum()),
            float(transformed.yMinimum()),
            float(transformed.xMaximum()),
            float(transformed.yMaximum()),
        )


def get_crs_backend() -> CrsBackend:
    """Return the active CRS backend."""
    if get_runtime().backend == "qgis":
        return QgisCrsBackend()
    return NativeCrsBackend()


def normalize_crs(crs: Any) -> Any:
    """Normalize CRS input with the active backend."""
    return get_crs_backend().normalize_crs(crs)


def crs_equal(left: Any, right: Any) -> bool:
    """Return whether two CRS values are equal."""
    if left is None or right is None:
        return left is right
    return get_crs_backend().crs_equal(left, right)


def crs_to_string(crs: Any) -> str:
    """Return a stable CRS string."""
    return get_crs_backend().crs_to_string(crs)


def transform_points(
    x_values: Any,
    y_values: Any,
    source_crs: Any,
    target_crs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform x/y coordinates with the active CRS backend."""
    return get_crs_backend().transform_points(
        x_values,
        y_values,
        source_crs,
        target_crs,
    )


def transform_bounds(
    bounds: BoundingBox,
    target_crs: Any,
) -> tuple[float, float, float, float]:
    """Transform bounds with the active CRS backend."""
    return get_crs_backend().transform_bounds(bounds, target_crs)
