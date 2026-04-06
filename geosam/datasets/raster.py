"""Raster dataset access built on top of :mod:`rasterio`."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from os import fspath
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from pyproj.crs import CRS
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform

from geosam.datasets.geogrid import GeoGrid
from geosam.logging import setup_logger
from geosam.query import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from geosam.typing import CrsLike, PathLike

logger = setup_logger(__name__)


def _normalize_raster_source(source_path: PathLike) -> str:
    """Normalize a raster source into a string accepted by rasterio.

    Parameters
    ----------
    source_path : PathLike
        Raster source path, URI, or GDAL-readable dataset string.

    Returns
    -------
    str
        Normalized source string.

    Notes
    -----
    Existing filesystem paths are resolved to absolute paths. Non-filesystem
    strings such as GDAL virtual filesystem paths or provider URIs are returned
    unchanged so callers can still rely on rasterio/GDAL support.

    """
    source_text = fspath(source_path)
    candidate_path = Path(source_text).expanduser()
    if candidate_path.exists():
        return str(candidate_path.resolve())
    return source_text


@dataclass(slots=True)
class RasterSample:
    """A raster chip read from a source dataset.

    Parameters
    ----------
    image : np.ndarray
        Array in ``(bands, height, width)`` order.
    bbox : BoundingBox
        Spatial extent of the returned chip.
    crs : CRS
        Coordinate reference system of the chip.
    transform : Affine
        Affine transform of the chip.
    shape : tuple[int, int]
        Chip shape as ``(height, width)``.
    source_path : str
        Source raster path.

    """

    image: np.ndarray
    bbox: BoundingBox
    crs: CRS
    transform: Affine
    shape: tuple[int, int]
    source_path: str

    @property
    def grid(self) -> GeoGrid:
        """Return the chip grid."""
        return GeoGrid(self.transform, self.shape, self.crs)

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary representation for collation."""
        return {
            "image": self.image,
            "bbox": self.bbox,
            "crs": self.crs,
            "transform": self.transform,
            "shape": self.shape,
            "source_path": self.source_path,
        }

    def to_model_image(
        self,
        *,
        pca: bool = False,
        value_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Convert the raster chip into an ``HWC`` uint8 image for SAM.

        Parameters
        ----------
        pca : bool, optional
            Whether to apply PCA band reduction when the raster contains more
            than three bands. If ``False``, the first three bands are used.
        value_range : tuple[float, float] | None, optional
            Fixed data range used to normalize all bands into ``uint8``. When
            omitted, each band is normalized independently from its finite
            minimum and maximum values.

        Returns
        -------
        np.ndarray
            Image array in ``(height, width, 3)`` order with ``uint8`` dtype.

        """
        image = self.image
        if image.ndim != 3:
            msg = f"RasterSample image must have 3 dimensions. Got {image.shape}."
            logger.error(msg)
            raise ValueError(msg)

        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        elif image.shape[0] == 2:
            image = np.concatenate([image, image[-1:, ...]], axis=0)
        elif image.shape[0] > 3:
            image = _pca_reduce_bands(image, n_components=3) if pca else image[:3, ...]

        normalized = np.stack(
            [
                _normalize_band_to_uint8(image[index], value_range=value_range)
                for index in range(image.shape[0])
            ],
            axis=0,
        )
        return np.moveaxis(normalized, 0, -1)


def _normalize_band_to_uint8(
    band: np.ndarray,
    *,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Normalize a single band into ``uint8`` range."""
    if band.dtype == np.uint8:
        return band

    band_float = band.astype(np.float32)
    finite_mask = np.isfinite(band_float)
    if not bool(np.any(finite_mask)):
        return np.zeros_like(band_float, dtype=np.uint8)

    finite_values = band_float[finite_mask]
    if value_range is None:
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
    else:
        min_value = float(value_range[0])
        max_value = float(value_range[1])
    if max_value == min_value:
        return np.zeros_like(band_float, dtype=np.uint8)

    scaled = (band_float - min_value) / (max_value - min_value)
    scaled = np.clip(scaled * 255.0, 0.0, 255.0)
    return scaled.astype(np.uint8)


def _pca_reduce_bands(image: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Reduce multi-band raster data with PCA.

    Parameters
    ----------
    image : np.ndarray
        Input image in ``(bands, height, width)`` order.
    n_components : int, optional
        Number of principal components to retain.

    Returns
    -------
    np.ndarray
        PCA-transformed image in ``(n_components, height, width)`` order.

    Raises
    ------
    ValueError
        If the input does not have three dimensions or if the number of bands
        is smaller than ``n_components``.

    """
    if image.ndim != 3:
        msg = f"PCA expects a 3D image tensor. Got {image.shape}."
        logger.error(msg)
        raise ValueError(msg)

    band_count, height, width = image.shape
    if band_count < n_components:
        msg = f"PCA requires at least {n_components} bands, but got {band_count}."
        logger.error(msg)
        raise ValueError(msg)

    flattened = image.astype(np.float32).reshape(band_count, height * width).T
    finite_mask = np.isfinite(flattened)
    if not bool(np.any(finite_mask)):
        return np.zeros((n_components, height, width), dtype=np.float32)

    column_means = np.zeros((flattened.shape[1],), dtype=np.float32)
    for index in range(flattened.shape[1]):
        valid_values = flattened[finite_mask[:, index], index]
        if valid_values.size == 0:
            column_means[index] = 0.0
        else:
            column_means[index] = float(np.mean(valid_values))
            flattened[~finite_mask[:, index], index] = column_means[index]

    centered = flattened - column_means
    _, _, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    principal_axes = right_singular_vectors[:n_components].T
    transformed = centered @ principal_axes
    return transformed.T.reshape(n_components, height, width)


class RasterDataset:
    """A single-raster dataset queried by :class:`BoundingBox`."""

    def __init__(
        self,
        source_path: PathLike,
        *,
        indexes: Sequence[int] | None = None,
        fill_value: float = 0,
        crs: CrsLike | None = None,
        res: float | tuple[float, float] | None = None,
        resampling: Resampling = Resampling.average,
    ) -> None:
        """Initialize a raster-backed dataset.

        Parameters
        ----------
        source_path : PathLike
            Path to the source raster.
        indexes : Sequence[int] | None, optional
            Band indexes to read. If omitted, all bands are used.
        fill_value : float, optional
            Fill value used for raster reads.
        crs : CrsLike | None, optional
            Output CRS for on-the-fly reprojection. If omitted, the source CRS
            is used.
        res : float | tuple[float, float] | None, optional
            Output pixel size in the output CRS. If omitted, the source or
            default warped resolution is used.
        resampling : Resampling, optional
            Default resampling method used for reads. Defaults to
            ``Resampling.average``.

        """
        self.source_path = _normalize_raster_source(source_path)
        self.fill_value = fill_value
        self.resampling = resampling
        self._vrt_options: dict[str, object] | None = None
        try:
            with rasterio.open(self.source_path) as dataset:
                self.source_crs = dataset.crs
                self.source_transform = dataset.transform
                self.source_shape = (dataset.height, dataset.width)
                self.source_bounds = BoundingBox(*dataset.bounds, crs=dataset.crs)
                self.count = dataset.count
                self.indexes = (
                    tuple(indexes)
                    if indexes is not None
                    else tuple(range(1, dataset.count + 1))
                )
                self._vrt_options = self._build_vrt_options(
                    dataset,
                    crs=crs,
                    res=res,
                )

                with self._open_reader(dataset=dataset) as reader:
                    self.crs = reader.crs
                    self.transform = reader.transform
                    self.shape = (reader.height, reader.width)
                    self.bounds = BoundingBox(*reader.bounds, crs=reader.crs)
        except rasterio.errors.RasterioIOError as exc:
            msg = f"Raster source could not be opened by rasterio: {self.source_path}"
            logger.exception(msg)
            raise FileNotFoundError(msg) from exc

        self.grid = GeoGrid(self.transform, self.shape, self.crs)

    @staticmethod
    def _normalize_res(
        res: float | tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        """Normalize output resolution to ``(x_res, y_res)``."""
        if res is None:
            return None
        if isinstance(res, (int, float)):
            return (abs(float(res)), abs(float(res)))
        return (abs(float(res[0])), abs(float(res[1])))

    def _build_vrt_options(
        self,
        dataset: rasterio.io.DatasetReader,
        *,
        crs: CrsLike | None,
        res: float | tuple[float, float] | None,
    ) -> dict[str, object] | None:
        """Build WarpedVRT options for output reprojection and resampling."""
        target_crs = dataset.crs if crs is None else CRS.from_user_input(crs)
        target_res = self._normalize_res(res)
        needs_vrt = target_crs != dataset.crs or target_res is not None
        if not needs_vrt:
            return None

        vrt_options: dict[str, object] = {
            "crs": target_crs,
            "resampling": self.resampling,
        }
        if target_res is not None:
            target_transform, target_width, target_height = calculate_default_transform(
                dataset.crs,
                target_crs,
                dataset.width,
                dataset.height,
                *dataset.bounds,
                resolution=target_res,
            )
            vrt_options.update({
                "transform": target_transform,
                "width": target_width,
                "height": target_height,
            })
        return vrt_options

    @contextmanager
    def _open_reader(
        self,
        *,
        dataset: rasterio.io.DatasetReader | None = None,
        resampling: Resampling | None = None,
    ) -> Iterator[rasterio.io.DatasetReader | WarpedVRT]:
        """Open a source dataset or warped VRT reader."""
        if dataset is not None:
            if self._vrt_options is None:
                yield dataset
                return

            vrt_options = dict(self._vrt_options)
            vrt_options["resampling"] = (
                self.resampling if resampling is None else resampling
            )
            with WarpedVRT(dataset, **vrt_options) as vrt:
                yield vrt
            return

        with (
            rasterio.open(self.source_path) as source_dataset,
            self._open_reader(
                dataset=source_dataset,
                resampling=resampling,
            ) as reader,
        ):
            yield reader

    def __len__(self) -> int:
        """Return the number of raster sources."""
        return 1

    def __getitem__(self, query: BoundingBox) -> RasterSample:
        """Read a chip intersecting the requested bounding box."""
        return self.read(query)

    def read(
        self,
        query: BoundingBox,
        *,
        out_shape: tuple[int, int] | None = None,
        resampling: Resampling | None = None,
    ) -> RasterSample:
        """Read data for a geographic bounding box."""
        projected = query if query.crs == self.crs else query.to_crs(self.crs)
        clipped = projected & self.bounds
        if clipped is None:
            msg = "Requested bounding box does not intersect the raster extent."
            logger.error(msg)
            raise ValueError(msg)

        with self._open_reader(resampling=resampling) as dataset:
            read_window = from_bounds(*clipped.to_tuple(), transform=dataset.transform)
            output_shape = None
            if out_shape is not None:
                output_shape = (len(self.indexes), int(out_shape[0]), int(out_shape[1]))
            image = dataset.read(
                self.indexes,
                window=read_window,
                out_shape=output_shape,
                resampling=self.resampling if resampling is None else resampling,
                boundless=False,
                fill_value=self.fill_value,
            )
            chip_transform = window_transform(read_window, dataset.transform)

        chip_height, chip_width = image.shape[-2:]
        chip_bbox = BoundingBox(
            *array_bounds(chip_height, chip_width, chip_transform),
            crs=self.crs,
        )
        return RasterSample(
            image=image,
            bbox=chip_bbox,
            crs=self.crs,
            transform=chip_transform,
            shape=(chip_height, chip_width),
            source_path=self.source_path,
        )

    def read_window(
        self,
        *,
        row_off: int,
        col_off: int,
        height: int,
        width: int,
        out_shape: tuple[int, int] | None = None,
        resampling: Resampling | None = None,
    ) -> RasterSample:
        """Read data for a pixel-aligned window."""
        if row_off < 0 or col_off < 0:
            msg = "Window offsets must be non-negative."
            logger.error(msg)
            raise ValueError(msg)
        if height <= 0 or width <= 0:
            msg = "Window shape must be positive."
            logger.error(msg)
            raise ValueError(msg)

        row_stop = min(row_off + height, self.shape[0])
        col_stop = min(col_off + width, self.shape[1])
        read_window = Window(
            col_off=col_off,
            row_off=row_off,
            width=col_stop - col_off,
            height=row_stop - row_off,
        )

        with self._open_reader(resampling=resampling) as dataset:
            output_shape = None
            if out_shape is not None:
                output_shape = (len(self.indexes), int(out_shape[0]), int(out_shape[1]))
            image = dataset.read(
                self.indexes,
                window=read_window,
                out_shape=output_shape,
                resampling=self.resampling if resampling is None else resampling,
                boundless=False,
                fill_value=self.fill_value,
            )
            chip_transform = window_transform(read_window, dataset.transform)

        chip_height, chip_width = image.shape[-2:]
        chip_bbox = BoundingBox(
            *array_bounds(chip_height, chip_width, chip_transform),
            crs=self.crs,
        )
        return RasterSample(
            image=image,
            bbox=chip_bbox,
            crs=self.crs,
            transform=chip_transform,
            shape=(chip_height, chip_width),
            source_path=self.source_path,
        )
