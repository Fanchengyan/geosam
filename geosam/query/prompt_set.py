"""Composite prompt queries for GeoSAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from geosam.crs import crs_equal
from geosam.logging import setup_logger
from geosam.query.bbox import BoundingBox
from geosam.query.points import Points

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from geosam.typing import CrsLike


@dataclass(frozen=True)
class PromptSet:
    """Composite prompt query combining points and an optional bounding box.

    Parameters
    ----------
    points : Points | None, optional
        Point prompts with optional foreground/background labels.
    bbox : BoundingBox | None, optional
        Bounding-box prompt.

    Raises
    ------
    ValueError
        If both ``points`` and ``bbox`` are missing or their coordinate systems
        cannot be resolved.

    Notes
    -----
    When both prompts carry a CRS and the CRS values differ, the bounding box is
    reprojected into the point CRS so the prompt set keeps a single shared CRS.

    """

    points: Optional[Points] = None
    bbox: Optional[BoundingBox] = None

    def __post_init__(self) -> None:
        """Validate prompt presence and normalize CRS handling."""
        if self.points is None and self.bbox is None:
            msg = "PromptSet requires at least one prompt."
            logger.error(msg)
            raise ValueError(msg)

        normalized_points = self.points
        normalized_bbox = self.bbox

        if normalized_points is not None and normalized_bbox is not None:
            if normalized_points.crs is None and normalized_bbox.crs is not None:
                normalized_points = Points(
                    normalized_points.values,
                    labels=normalized_points.labels,
                    crs=normalized_bbox.crs,
                )
            elif normalized_bbox.crs is None and normalized_points.crs is not None:
                normalized_bbox = BoundingBox(
                    *normalized_bbox.to_tuple(),
                    crs=normalized_points.crs,
                )
            elif (
                normalized_points.crs is not None
                and normalized_bbox.crs is not None
                and not crs_equal(normalized_points.crs, normalized_bbox.crs)
            ):
                normalized_bbox = normalized_bbox.to_crs(normalized_points.crs)

        object.__setattr__(self, "points", normalized_points)
        object.__setattr__(self, "bbox", normalized_bbox)

    @property
    def crs(self) -> Any | None:
        """Return the shared CRS of the prompt set."""
        if self.points is not None and self.points.crs is not None:
            return self.points.crs
        if self.bbox is not None:
            return self.bbox.crs
        return None

    @property
    def bounds(self) -> BoundingBox:
        """Return the combined bounds of all prompts."""
        if self.points is not None and self.bbox is not None:
            return self.points.bounds | self.bbox
        if self.points is not None:
            return self.points.bounds
        if self.bbox is not None:
            return self.bbox
        msg = "PromptSet requires at least one prompt."
        logger.error(msg)
        raise ValueError(msg)

    @property
    def center(self) -> tuple[float, float]:
        """Return the center of the combined prompt bounds."""
        return self.bounds.center

    @property
    def has_points(self) -> bool:
        """Return whether point prompts are present."""
        return self.points is not None

    @property
    def has_bbox(self) -> bool:
        """Return whether a bounding-box prompt is present."""
        return self.bbox is not None

    def to_crs(self, crs: CrsLike) -> PromptSet:
        """Reproject the prompt set into another CRS.

        Parameters
        ----------
        crs : CRS | str
            Destination CRS.

        Returns
        -------
        PromptSet
            Reprojected prompt set.

        Raises
        ------
        ValueError
            If the prompt set has no source CRS.

        """
        if self.crs is None:
            msg = "Cannot reproject a PromptSet without a source CRS."
            logger.error(msg)
            raise ValueError(msg)

        return PromptSet(
            points=None if self.points is None else self.points.to_crs(crs),
            bbox=None if self.bbox is None else self.bbox.to_crs(crs),
        )
