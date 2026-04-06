"""Query normalization and prompt-conversion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from geosam.logging import setup_logger
from geosam.query.bbox import BoundingBox
from geosam.query.points import Points
from geosam.query.prompt_set import PromptSet

if TYPE_CHECKING:
    from geosam.datasets.geogrid import GeoGrid

logger = setup_logger(__name__)

GeoQuery: TypeAlias = BoundingBox | Points | PromptSet


def normalize_chip_size(chip_size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize chip size to ``(height, width)``."""
    if isinstance(chip_size, int):
        return (chip_size, chip_size)
    return (int(chip_size[0]), int(chip_size[1]))


def query_bounds(query: GeoQuery) -> BoundingBox:
    """Return the geometric bounds of a query."""
    if isinstance(query, PromptSet):
        return query.bounds
    if isinstance(query, BoundingBox):
        return query
    return query.bounds


def query_center(query: GeoQuery) -> tuple[float, float]:
    """Return the geometric center of a query."""
    if isinstance(query, PromptSet):
        return query.center
    if isinstance(query, BoundingBox):
        return query.center
    return query.center


def window_from_center(
    center: tuple[float, float],
    chip_size: int | tuple[int, int],
    *,
    grid: GeoGrid,
) -> BoundingBox:
    """Return an aligned chip window around a query center."""
    chip_height, chip_width = normalize_chip_size(chip_size)
    if grid.height <= chip_height and grid.width <= chip_width:
        return grid.bounds

    row_values, col_values = grid.row_col([center[0]], [center[1]], op=float)
    center_row = float(row_values[0])
    center_col = float(col_values[0])
    row_start = int(center_row - chip_height / 2)
    col_start = int(center_col - chip_width / 2)

    max_row_start = max(grid.height - chip_height, 0)
    max_col_start = max(grid.width - chip_width, 0)
    row_start = min(max(row_start, 0), max_row_start)
    col_start = min(max(col_start, 0), max_col_start)

    return grid.window(
        row_off=row_start,
        col_off=col_start,
        height=min(chip_height, grid.height),
        width=min(chip_width, grid.width),
    ).bounds


def points_to_prompt(
    points: Points,
    chip_grid: GeoGrid,
    dst_shape: tuple[int, int] | None = None,
    *,
    strict: bool = True,
) -> tuple[object, object]:
    """Convert point queries into SAM pixel prompts."""
    return chip_grid.to_points_prompt(points, dst_shape=dst_shape, strict=strict)


def bbox_to_prompt(
    bbox: BoundingBox,
    chip_grid: GeoGrid,
    dst_shape: tuple[int, int] | None = None,
    *,
    strict: bool = True,
) -> tuple[float, float, float, float]:
    """Convert a bounding-box query into a SAM pixel prompt."""
    return chip_grid.to_bbox_prompt(bbox, dst_shape=dst_shape, strict=strict)
