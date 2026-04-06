"""Spatial bounding-box utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, overload

import geopandas as gpd
from pyproj.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box

from geosam.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from shapely.geometry.base import BaseGeometry

    from geosam.typing import CrsLike

logger = setup_logger(__name__)


class BoundingBox:
    """A spatial bounding box with an optional CRS.

    Parameters
    ----------
    left : float
        Western boundary.
    bottom : float
        Southern boundary.
    right : float
        Eastern boundary.
    top : float
        Northern boundary.
    crs : CrsLike | None, optional
        Coordinate reference system of the bounds.

    """

    __slots__ = ["_crs", "bottom", "left", "right", "top"]
    __hash__ = object.__hash__

    def __init__(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
        crs: Optional[CrsLike] = None,
    ) -> None:
        """Initialize a bounding box."""
        if left > right:
            msg = f"Invalid bounding box: left={left} is greater than right={right}."
            logger.error(msg)
            raise ValueError(msg)
        if bottom > top:
            msg = f"Invalid bounding box: bottom={bottom} is greater than top={top}."
            logger.error(msg)
            raise ValueError(msg)

        self.left = float(left)
        self.bottom = float(bottom)
        self.right = float(right)
        self.top = float(top)
        self._crs = CRS.from_user_input(crs) if crs is not None else None

    def __repr__(self) -> str:
        """Return the representation of the bounding box."""
        return (
            "BoundingBox("
            f"left={self.left}, bottom={self.bottom}, "
            f"right={self.right}, top={self.top}, crs={self.crs})"
        )

    def __str__(self) -> str:
        """Return the string representation of the bounding box."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Compare two bounding boxes."""
        if not isinstance(other, BoundingBox):
            return False
        return (
            self.left == other.left
            and self.bottom == other.bottom
            and self.right == other.right
            and self.top == other.top
            and self.crs == other.crs
        )

    @overload
    def __getitem__(self, key: int) -> float: ...

    @overload
    def __getitem__(self, key: slice) -> list[float]: ...

    def __getitem__(self, key: Union[int, slice]) -> Union[float, list[float]]:
        """Index the ``(left, bottom, right, top)`` tuple."""
        return [self.left, self.bottom, self.right, self.top][key]

    def __iter__(self) -> Iterator[float]:
        """Iterate over bounds in ``(left, bottom, right, top)`` order."""
        yield from [self.left, self.bottom, self.right, self.top]

    def __contains__(self, other: BoundingBox) -> bool:
        """Return whether ``other`` is fully contained in this box."""
        return self.contains(other)

    def __or__(self, other: BoundingBox) -> BoundingBox:
        """Return the union of two bounding boxes."""
        return self.union(other)

    def __and__(self, other: BoundingBox) -> Optional[BoundingBox]:
        """Return the intersection of two bounding boxes."""
        return self.intersection(other)

    @property
    def crs(self) -> Optional[CRS]:
        """Coordinate reference system of the bounding box."""
        return self._crs

    @property
    def area(self) -> float:
        """Spatial area covered by the bounding box."""
        return (self.right - self.left) * (self.top - self.bottom)

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        return ((self.left + self.right) / 2.0, (self.bottom + self.top) / 2.0)

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.top - self.bottom

    def _ensure_shared_crs(
        self, other: BoundingBox
    ) -> tuple[BoundingBox, Optional[CRS]]:
        """Convert the other bounding box to the same CRS when possible."""
        if self.crs == other.crs:
            return other, self.crs

        if self.crs is None or other.crs is None:
            crs_new = self.crs or other.crs
            logger.warning(
                "Bounding-box CRS is missing. Assuming both bounding boxes are "
                "already expressed in the same coordinate system."
            )
            return other, crs_new

        return other.to_crs(self.crs), self.crs

    def set_crs(self, crs: Union[CRS, str]) -> None:
        """Assign a CRS without reprojection."""
        self._crs = CRS.from_user_input(crs)

    def to_crs(self, crs: Union[CRS, str]) -> BoundingBox:
        """Reproject the bounding box."""
        if self.crs is None:
            msg = "Cannot reproject a bounding box without a source CRS."
            logger.error(msg)
            raise ValueError(msg)

        target_crs = CRS.from_user_input(crs)
        if self.crs == target_crs:
            return self

        left, bottom, right, top = transform_bounds(
            self.crs,
            target_crs,
            self.left,
            self.bottom,
            self.right,
            self.top,
        )
        return BoundingBox(left, bottom, right, top, crs=target_crs)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(left, bottom, right, top)``."""
        return (self.left, self.bottom, self.right, self.top)

    def to_dict(self) -> dict[str, float]:
        """Return a dictionary representation."""
        return {
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
            "top": self.top,
        }

    def to_geometry(self) -> BaseGeometry:
        """Convert the bounding box to a Shapely geometry."""
        return box(self.left, self.bottom, self.right, self.top)

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert the bounding box to a GeoDataFrame."""
        return gpd.GeoDataFrame(geometry=[self.to_geometry()], crs=self.crs)

    def union(self, other: BoundingBox) -> BoundingBox:
        """Return the minimum box covering both bounding boxes."""
        other, crs_new = self._ensure_shared_crs(other)
        return BoundingBox(
            min(self.left, other.left),
            min(self.bottom, other.bottom),
            max(self.right, other.right),
            max(self.top, other.top),
            crs=crs_new,
        )

    def intersection(self, other: BoundingBox) -> Optional[BoundingBox]:
        """Return the overlapping region of two bounding boxes."""
        other, crs_new = self._ensure_shared_crs(other)
        if not self.intersects(other):
            logger.warning("Bounding boxes %s and %s do not overlap.", self, other)
            return None

        return BoundingBox(
            max(self.left, other.left),
            max(self.bottom, other.bottom),
            min(self.right, other.right),
            min(self.top, other.top),
            crs=crs_new,
        )

    def intersects(self, other: BoundingBox) -> bool:
        """Return whether two bounding boxes overlap."""
        other, _ = self._ensure_shared_crs(other)
        return (
            self.left <= other.right
            and self.right >= other.left
            and self.bottom <= other.top
            and self.top >= other.bottom
        )

    def contains(self, other: BoundingBox) -> bool:
        """Return whether the bounding box fully contains another box."""
        other, _ = self._ensure_shared_crs(other)
        return (
            self.left <= other.left
            and self.right >= other.right
            and self.bottom <= other.bottom
            and self.top >= other.top
        )

    def split(
        self,
        proportion: float,
        horizontal: bool = True,
    ) -> tuple[BoundingBox, BoundingBox]:
        """Split the bounding box in two."""
        if not 0.0 < proportion < 1.0:
            msg = "Split proportion must be between 0 and 1."
            logger.error(msg)
            raise ValueError(msg)

        if horizontal:
            split_x = self.left + self.width * proportion
            return (
                BoundingBox(self.left, self.bottom, split_x, self.top, crs=self.crs),
                BoundingBox(split_x, self.bottom, self.right, self.top, crs=self.crs),
            )

        split_y = self.bottom + self.height * proportion
        return (
            BoundingBox(self.left, self.bottom, self.right, split_y, crs=self.crs),
            BoundingBox(self.left, split_y, self.right, self.top, crs=self.crs),
        )

    def buffer(self, distance: float) -> BoundingBox:
        """Return a buffered bounding box."""
        return BoundingBox(
            self.left - distance,
            self.bottom - distance,
            self.right + distance,
            self.top + distance,
            crs=self.crs,
        )
