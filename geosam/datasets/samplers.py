"""Grid-based samplers for raster chips."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from geosam.logging import setup_logger
from geosam.query.prompts import normalize_chip_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from geosam.datasets.raster import RasterDataset
    from geosam.query import BoundingBox

logger = setup_logger(__name__)


def _normalize_overlap(
    overlap: Optional[Union[int, tuple[int, int]]],
) -> tuple[int, int]:
    """Normalize overlap input."""
    if overlap is None:
        return (0, 0)
    return normalize_chip_size(overlap)


def _window_starts(size: int, chip_size: int, stride: int) -> list[int]:
    """Return deterministic sliding-window start indices."""
    if chip_size >= size:
        return [0]

    starts = list(range(0, max(size - chip_size, 0) + 1, stride))
    last_start = size - chip_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


class GridGeoSampler:
    """Generate geographic chip bounds from a raster dataset."""

    def __init__(
        self,
        dataset: RasterDataset,
        *,
        chip_size: Union[int, tuple[int, int]],
        stride: Optional[Union[int, tuple[int, int]]] = None,
        overlap: Optional[Union[int, tuple[int, int]]] = None,
    ) -> None:
        """Initialize a grid sampler."""
        self.dataset = dataset
        self.chip_size = normalize_chip_size(chip_size)
        overlap_shape = _normalize_overlap(overlap)
        if stride is None:
            self.stride = (
                self.chip_size[0] - overlap_shape[0],
                self.chip_size[1] - overlap_shape[1],
            )
        else:
            self.stride = normalize_chip_size(stride)

        if self.stride[0] <= 0 or self.stride[1] <= 0:
            msg = "Sampler stride must be positive after overlap adjustment."
            logger.error(msg)
            raise ValueError(msg)

    def __iter__(self) -> Iterator[BoundingBox]:
        """Yield chip bounds in row-major order."""
        chip_height, chip_width = self.chip_size
        row_starts = _window_starts(self.dataset.shape[0], chip_height, self.stride[0])
        col_starts = _window_starts(self.dataset.shape[1], chip_width, self.stride[1])

        for row_start in row_starts:
            for col_start in col_starts:
                yield self.dataset.grid.window(
                    row_off=row_start,
                    col_off=col_start,
                    height=min(chip_height, self.dataset.shape[0]),
                    width=min(chip_width, self.dataset.shape[1]),
                ).bounds

    def __len__(self) -> int:
        """Return the number of sampled chips."""
        chip_height, chip_width = self.chip_size
        row_starts = _window_starts(self.dataset.shape[0], chip_height, self.stride[0])
        col_starts = _window_starts(self.dataset.shape[1], chip_width, self.stride[1])
        return len(row_starts) * len(col_starts)
