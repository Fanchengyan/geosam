"""GeoGrid utilities for raster metadata and spatial-to-pixel conversion."""

from __future__ import annotations

import pprint
from typing import TYPE_CHECKING, Literal

import numpy as np
from pyproj.crs import CRS
from rasterio import Affine, transform, windows
from rasterio.transform import array_bounds, rowcol, xy
from rasterio.warp import calculate_default_transform

from geosam.logging import setup_logger
from geosam.query import BoundingBox, Points

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    import numpy.typing as npt
    from rasterio.io import DatasetReader

    from geosam.typing import CrsLike

logger = setup_logger(__name__)

OFFSET_LOCATIONS: dict[
    Literal["center", "ul", "ur", "ll", "lr"], tuple[float, float]
] = {
    "center": (0.5, 0.5),
    "ul": (0.0, 0.0),
    "ur": (1.0, 0.0),
    "ll": (0.0, 1.0),
    "lr": (1.0, 1.0),
}


def _offset_from_loc(
    loc: Literal["center", "ul", "ur", "ll", "lr"],
) -> tuple[float, float]:
    """Return fractional pixel offsets for a pixel location."""
    if loc not in OFFSET_LOCATIONS:
        msg = f"Unsupported pixel location {loc!r}."
        logger.error(msg)
        raise ValueError(msg)
    return OFFSET_LOCATIONS[loc]


def _normalize_shape(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a shape-like input to ``(height, width)``."""
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


class GeoGridMixin:
    """Common metadata accessors shared by geogrid-like objects."""

    def _refresh_bounds(self) -> None:
        """Refresh cached bounds after geometry changes."""
        if (
            hasattr(self, "_transform")
            and hasattr(self, "_shape")
            and hasattr(self, "_crs")
        ):
            self._bounds = self._parse_bounds()

    def _parse_bounds(self) -> BoundingBox:
        """Parse bounds from the transform and shape."""
        west, south, east, north = array_bounds(self.height, self.width, self.transform)
        return BoundingBox(west, south, east, north, crs=self.crs)

    @property
    def transform(self) -> Affine:
        """Affine transform mapping pixel to world coordinates."""
        return self._transform

    @transform.setter
    def transform(self, value: Affine) -> None:
        """Set the affine transform."""
        self._transform = value
        self._refresh_bounds()

    @property
    def shape(self) -> tuple[int, int]:
        """Return raster shape as ``(height, width)``."""
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int, int]) -> None:
        """Set raster shape."""
        if len(value) != 2:
            msg = "GeoGrid shape must be a 2-tuple of (height, width)."
            logger.error(msg)
            raise ValueError(msg)
        self._shape = (int(value[0]), int(value[1]))
        self._refresh_bounds()

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the grid."""
        return self._crs

    @crs.setter
    def crs(self, value: CrsLike) -> None:
        """Set the coordinate reference system."""
        self._crs = CRS.from_user_input(value)
        self._refresh_bounds()

    @property
    def width(self) -> int:
        """Raster width in pixels."""
        return self.shape[1]

    @property
    def height(self) -> int:
        """Raster height in pixels."""
        return self.shape[0]

    @property
    def res(self) -> tuple[float, float]:
        """Pixel resolution in x/y order."""
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> BoundingBox:
        """Grid bounds in the grid CRS."""
        return self._bounds

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return ``(left, right, bottom, top)`` extent."""
        bounds = self.bounds
        return (bounds.left, bounds.right, bounds.bottom, bounds.top)


class GeoGrid(GeoGridMixin):
    """A fixed pixel grid reference system for geospatial rasters."""

    def __init__(
        self,
        transform: Affine,
        shape: tuple[int, int],
        crs: CrsLike,
    ) -> None:
        """Initialize a GeoGrid."""
        self._transform = transform
        self.crs = crs
        self.shape = shape

    def __repr__(self) -> str:
        """Return a readable GeoGrid representation."""
        info = {
            "bounds": self.bounds.to_tuple(),
            "transform": tuple(self.transform),
            "shape": self.shape,
            "crs": self.crs.to_string(),
        }
        repr_str = f" {pprint.pformat(info, indent=2, sort_dicts=False).strip('{}')}"
        return f"GeoGrid(\n{repr_str}\n)"

    @classmethod
    def from_dataset(cls, dataset: DatasetReader) -> GeoGrid:
        """Build a GeoGrid from a rasterio dataset."""
        return cls(dataset.transform, (dataset.height, dataset.width), dataset.crs)

    @classmethod
    def from_bounds(
        cls,
        bounds: BoundingBox | tuple[float, float, float, float],
        *,
        res: float | tuple[float, float] | None = None,
        shape: tuple[int, int] | None = None,
        crs: CrsLike | None = None,
    ) -> Self:
        """Build a GeoGrid from bounds and either resolution or shape."""
        bounds, resolved_crs = format_bounds_and_crs(bounds, crs)
        left, bottom, right, top = bounds
        if res is not None:
            if isinstance(res, (int, float)):
                res = (float(res), float(res))
            xsize = abs(float(res[0]))
            ysize = abs(float(res[1]))
            width = int(np.ceil((right - left) / xsize))
            height = int(np.ceil((top - bottom) / ysize))
            shape = (height, width)
        elif shape is not None:
            width, height = shape[1], shape[0]
            xsize = (right - left) / width
            ysize = (top - bottom) / height
        else:
            msg = "GeoGrid.from_bounds requires either res or shape."
            logger.error(msg)
            raise ValueError(msg)

        return cls(transform.from_origin(left, top, xsize, ysize), shape, resolved_crs)

    @classmethod
    def from_xy(
        cls,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        crs: CrsLike = "WGS84",
        loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
    ) -> Self:
        """Build a GeoGrid from x/y coordinate vectors."""
        bounds, affine_transform, _, shape = geoinfo_from_xy(x, y, crs=crs, loc=loc)
        return cls(affine_transform, shape, bounds.crs)

    def to_crs(
        self,
        crs: CrsLike,
        *,
        res: float | tuple[float, float] | None = None,
        shape: tuple[int, int] | None = None,
    ) -> GeoGrid:
        """Return a reprojected GeoGrid."""
        left, bottom, right, top = self.bounds
        dst_width = None if shape is None else shape[1]
        dst_height = None if shape is None else shape[0]
        affine_transform, width, height = calculate_default_transform(
            self.crs,
            crs,
            self.width,
            self.height,
            left,
            bottom,
            right,
            top,
            dst_width=dst_width,
            dst_height=dst_height,
            resolution=res,
        )
        return GeoGrid(affine_transform, (height, width), crs)

    def window(self, row_off: int, col_off: int, height: int, width: int) -> GeoGrid:
        """Return a sub-grid for a pixel window."""
        if row_off < 0 or col_off < 0:
            msg = "Window offsets must be non-negative."
            logger.error(msg)
            raise ValueError(msg)
        if height <= 0 or width <= 0:
            msg = "Window shape must be positive."
            logger.error(msg)
            raise ValueError(msg)
        if row_off + height > self.height or col_off + width > self.width:
            msg = "Requested window exceeds the GeoGrid extent."
            logger.error(msg)
            raise ValueError(msg)

        window = windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=width,
            height=height,
        )
        return GeoGrid(
            windows.transform(window, self.transform),
            (height, width),
            self.crs,
        )

    def to_view(self, roi: BoundingBox) -> GeoGrid:
        """Return a new GeoGrid aligned to a region of interest."""
        roi, _ = format_bounds_and_crs(roi, self.crs)
        if roi == self.bounds:
            return self

        row_values, col_values = rowcol(
            self.transform,
            [roi.left, roi.right, roi.right, roi.left],
            [roi.top, roi.top, roi.bottom, roi.bottom],
            op=float,
        )
        row_start = int(np.floor(min(row_values)))
        row_stop = int(np.ceil(max(row_values)))
        col_start = int(np.floor(min(col_values)))
        col_stop = int(np.ceil(max(col_values)))
        row_start = max(row_start, 0)
        col_start = max(col_start, 0)
        row_stop = min(row_stop, self.height)
        col_stop = min(col_stop, self.width)
        return self.window(
            row_off=row_start,
            col_off=col_start,
            height=row_stop - row_start,
            width=col_stop - col_start,
        )

    def scale_pixel_coordinates(
        self,
        coordinates: np.ndarray,
        dst_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Scale pixel coordinates from source shape into another destination shape."""
        if dst_shape is None or tuple(dst_shape) == self.shape:
            return coordinates.astype(np.float32)

        dst_height, dst_width = dst_shape
        scaled = coordinates.astype(np.float32).copy()
        scaled[..., 0] *= dst_width / self.width
        scaled[..., 1] *= dst_height / self.height
        return scaled

    def _validate_point_prompt(self, coordinates: np.ndarray, strict: bool) -> None:
        """Validate point prompt coordinates against grid bounds."""
        if not strict:
            return

        within_x = np.logical_and(
            coordinates[:, 0] >= 0.0,
            coordinates[:, 0] <= self.width,
        )
        within_y = np.logical_and(
            coordinates[:, 1] >= 0.0,
            coordinates[:, 1] <= self.height,
        )
        if not bool(np.all(within_x & within_y)):
            msg = "Point prompt lies outside the chip bounds."
            logger.error(msg)
            raise ValueError(msg)

    def _validate_bbox_prompt(
        self,
        coordinates: tuple[float, float, float, float],
        strict: bool,
    ) -> None:
        """Validate bounding-box prompt coordinates against grid bounds."""
        if not strict:
            return

        x1, y1, x2, y2 = coordinates
        if x1 < 0.0 or y1 < 0.0 or x2 > self.width or y2 > self.height:
            msg = "Bounding-box prompt lies outside the chip bounds."
            logger.error(msg)
            raise ValueError(msg)

    def to_points_prompt(
        self,
        points: Points,
        dst_shape: tuple[int, int] | None = None,
        *,
        strict: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Convert geographic points into pixel prompt coordinates."""
        projected = points if points.crs == self.crs else points.to_crs(self.crs)
        row_values, col_values = self.row_col(projected.x, projected.y, op=float)
        coordinates = np.column_stack([col_values, row_values]).astype(np.float32)
        self._validate_point_prompt(coordinates, strict=strict)
        coordinates = self.scale_pixel_coordinates(coordinates, dst_shape=dst_shape)
        return coordinates, projected.labels

    def to_bbox_prompt(
        self,
        bbox: BoundingBox,
        dst_shape: tuple[int, int] | None = None,
        *,
        strict: bool = True,
    ) -> tuple[float, float, float, float]:
        """Convert a geographic bounding box into a pixel ``xyxy`` prompt."""
        projected = bbox if bbox.crs == self.crs else bbox.to_crs(self.crs)
        row_values, col_values = self.row_col(
            [projected.left, projected.right, projected.right, projected.left],
            [projected.top, projected.top, projected.bottom, projected.bottom],
            op=float,
        )
        coordinates = np.array(
            [
                float(np.min(col_values)),
                float(np.min(row_values)),
                float(np.max(col_values)),
                float(np.max(row_values)),
            ],
            dtype=np.float32,
        )
        scaled = self.scale_pixel_coordinates(
            coordinates.reshape(2, 2),
            dst_shape=dst_shape,
        ).reshape(-1)
        scaled_tuple = tuple(float(value) for value in scaled)
        self._validate_bbox_prompt(scaled_tuple, strict=strict)
        return scaled_tuple

    def get_xy(
        self,
        loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return x/y coordinate vectors for the grid."""
        return xy_from_transform(self.transform, self.width, self.height, loc=loc)

    def row_col(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        op: Callable | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert world coordinates to row/column indices."""
        row_values, col_values = rowcol(self.transform, x, y, op=op)
        return np.asarray(row_values), np.asarray(col_values)

    def xy(
        self,
        row: npt.ArrayLike,
        col: npt.ArrayLike,
        offset: Literal["center", "ul", "ur", "ll", "lr"] = "center",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert row/column indices to world coordinates."""
        x_values, y_values = xy(self.transform, row, col, offset=offset)
        return np.asarray(x_values), np.asarray(y_values)


def xy_from_transform(
    affine_transform: Affine | None,
    width: int,
    height: int,
    *,
    loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinate vectors for a transform and shape."""
    if affine_transform is None:
        return np.arange(width), np.arange(height)
    offset = _offset_from_loc(loc)
    x_values = affine_transform.xoff + affine_transform.a * (
        np.arange(width) + offset[0]
    )
    y_values = affine_transform.yoff + affine_transform.e * (
        np.arange(height) + offset[1]
    )
    return x_values, y_values


def format_bounds_and_crs(
    bounds: BoundingBox | tuple[float, float, float, float],
    crs: CrsLike | None,
) -> tuple[BoundingBox, CRS]:
    """Normalize bounds and CRS input."""
    if not isinstance(bounds, BoundingBox):
        left, bottom, right, top = bounds
        bounds = BoundingBox(left, bottom, right, top, crs=crs)

    if crs is not None:
        resolved_crs = CRS.from_user_input(crs)
        if bounds.crs is not None and bounds.crs != resolved_crs:
            bounds = bounds.to_crs(resolved_crs)
        return bounds, resolved_crs

    if bounds.crs is None:
        msg = "A CRS is required when bounds do not carry one."
        logger.error(msg)
        raise ValueError(msg)
    return bounds, bounds.crs


def geoinfo_from_xy(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    crs: CrsLike,
    loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
) -> tuple[BoundingBox, Affine, tuple[float, float], tuple[int, int]]:
    """Infer a GeoGrid definition from x/y coordinate vectors."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1:
        msg = "x and y coordinates must be one-dimensional."
        logger.error(msg)
        raise ValueError(msg)
    if len(x_values) < 2 or len(y_values) < 2:
        msg = "At least two x and y coordinates are required."
        logger.error(msg)
        raise ValueError(msg)

    x_res = float(np.mean(np.diff(x_values)))
    y_res = float(np.mean(np.diff(y_values)))
    offset_x, offset_y = _offset_from_loc(loc)
    left = float(x_values[0] - x_res * offset_x)
    top = float(y_values[0] - y_res * offset_y)
    affine_transform = Affine.translation(left, top) * Affine.scale(x_res, y_res)
    shape = (len(y_values), len(x_values))
    bounds = BoundingBox(*array_bounds(shape[0], shape[1], affine_transform), crs=crs)
    return bounds, affine_transform, (abs(x_res), abs(y_res)), shape
