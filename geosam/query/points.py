"""Point-query utilities for geospatial prompts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyproj.crs import CRS

from geosam.logging import setup_logger
from geosam.query.bbox import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy.typing as npt

    from geosam.typing import CrsLike, PathLike

logger = setup_logger(__name__)

PointPromptLabel = Literal[0, 1]


class Points:
    """A collection of geospatial points with optional SAM labels."""

    __slots__ = ["_crs", "_labels", "_values"]

    _values: np.ndarray
    _labels: np.ndarray | None
    _crs: CRS | None

    def __init__(
        self,
        points: Sequence[float | Sequence[float]] | npt.ArrayLike,
        labels: (
            PointPromptLabel | Sequence[PointPromptLabel] | npt.ArrayLike | None
        ) = None,
        crs: CrsLike | None = None,
        dtype: npt.DTypeLike = np.float32,
        label_dtype: npt.DTypeLike = np.int8,
    ) -> None:
        """Initialize a collection of points."""
        values = np.asarray(points, dtype=dtype)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 2:
            msg = f"Points must have shape (n, 2). Got {values.shape}."
            logger.error(msg)
            raise ValueError(msg)

        self._values = values
        self._labels = self._normalize_labels(
            labels,
            point_count=len(values),
            dtype=label_dtype,
        )
        self._crs = CRS.from_user_input(crs) if crs is not None else None

    def __len__(self) -> int:
        """Return the number of points."""
        return int(self._values.shape[0])

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over points."""
        yield from self._values

    def __getitem__(self, key: int) -> Points:
        """Return a single point selection."""
        labels = None if self.labels is None else self.labels[key]
        return Points(self._values[key, :], labels=labels, crs=self.crs)

    def __contains__(self, item: Points | Sequence[float]) -> bool:
        """Return whether the given point or points are present."""
        candidate = item.values if isinstance(item, Points) else np.asarray(item)
        if candidate.ndim == 1:
            candidate = candidate.reshape(1, -1)
        if candidate.ndim != 2 or candidate.shape[1] != 2:
            msg = f"Candidate points must have shape (n, 2). Got {candidate.shape}."
            logger.error(msg)
            raise ValueError(msg)

        matches = np.any(
            np.all(
                self._values[:, np.newaxis, :] == candidate[np.newaxis, :, :],
                axis=2,
            ),
            axis=0,
        )
        return bool(np.all(matches))

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Return the underlying NumPy array."""
        if dtype is None:
            return self._values
        return self._values.astype(dtype)

    def __repr__(self) -> str:
        """Return a representation of the points."""
        return f"Points(count={len(self)}, crs={self.crs}, has_label={self.has_label})"

    def __str__(self) -> str:
        """Return a string representation of the points."""
        return self.__repr__()

    def __add__(self, other: Points) -> Points:
        """Concatenate two point collections."""
        if not isinstance(other, Points):
            msg = f"Expected Points, got {type(other)!r}."
            logger.error(msg)
            raise TypeError(msg)

        other, crs_new = self._ensure_shared_crs(other)
        if self.has_label != other.has_label:
            msg = "Cannot merge labeled points with unlabeled points."
            logger.error(msg)
            raise ValueError(msg)

        labels = None
        if self.labels is not None and other.labels is not None:
            labels = np.concatenate([self.labels, other.labels])
        return Points(
            np.vstack([self.values, other.values]),
            labels=labels,
            crs=crs_new,
        )

    def __sub__(self, other: Points) -> Points | None:
        """Subtract another point collection."""
        if not isinstance(other, Points):
            msg = f"Expected Points, got {type(other)!r}."
            logger.error(msg)
            raise TypeError(msg)

        other, crs_new = self._ensure_shared_crs(other)
        mask = ~np.any(
            np.all(
                self._values[:, np.newaxis, :] == other.values[np.newaxis, :, :],
                axis=2,
            ),
            axis=1,
        )
        values = self._values[mask]
        if len(values) == 0:
            return None
        labels = None if self.labels is None else self.labels[mask]
        return Points(values, labels=labels, crs=crs_new)

    @staticmethod
    def _normalize_labels(
        labels: PointPromptLabel | Sequence[PointPromptLabel] | npt.ArrayLike | None,
        point_count: int,
        dtype: npt.DTypeLike,
    ) -> np.ndarray | None:
        """Normalize labels to a 1D integer array."""
        if labels is None:
            return None

        normalized = np.asarray(labels, dtype=dtype)
        if normalized.ndim == 0:
            normalized = np.full(point_count, normalized.item(), dtype=dtype)
        elif normalized.ndim != 1:
            msg = f"Labels must be scalar or 1D. Got shape {normalized.shape}."
            logger.error(msg)
            raise ValueError(msg)
        elif normalized.shape[0] == 1 and point_count > 1:
            normalized = np.full(point_count, normalized[0], dtype=dtype)
        elif normalized.shape[0] != point_count:
            msg = (
                "Labels length must match the number of points. "
                f"Got {normalized.shape[0]} labels for {point_count} points."
            )
            logger.error(msg)
            raise ValueError(msg)

        if not np.isin(normalized, [0, 1]).all():
            msg = f"Point labels must be 0 or 1. Got {normalized.tolist()}."
            logger.error(msg)
            raise ValueError(msg)
        return normalized

    @staticmethod
    def _find_field(
        frame: gpd.GeoDataFrame | pd.DataFrame,
        field_names: list[str],
    ) -> str | None:
        """Return the first matching field name."""
        lowered = set(field_names)
        for name in frame.columns:
            if name.lower() in lowered:
                return name
        return None

    @classmethod
    def _ensure_fields(
        cls,
        frame: gpd.GeoDataFrame | pd.DataFrame,
        x_field: str | None,
        y_field: str | None,
    ) -> tuple[str, str]:
        """Resolve x/y coordinate field names."""
        if x_field is None:
            x_field = cls._find_field(
                frame,
                ["x", "xs", "lon", "longitude", "long", "longitudes"],
            )
        if y_field is None:
            y_field = cls._find_field(
                frame,
                ["y", "ys", "lat", "latitude", "lats", "latitudes"],
            )
        if x_field is None or y_field is None:
            msg = "Could not infer x/y fields from the input table."
            logger.error(msg)
            raise ValueError(msg)
        return x_field, y_field

    @classmethod
    def _resolve_labels(
        cls,
        frame: gpd.GeoDataFrame | pd.DataFrame,
        label_field: str | None,
        default_label: PointPromptLabel | None,
    ) -> np.ndarray | None:
        """Resolve labels from a tabular source."""
        if label_field is None:
            label_field = cls._find_field(
                frame,
                ["label", "labels", "point_label", "point_labels", "prompt_label"],
            )

        if label_field is not None:
            if label_field not in frame.columns:
                msg = f"Cannot find label field {label_field!r}."
                logger.error(msg)
                raise ValueError(msg)
            return frame[label_field].to_numpy(dtype=np.int8)

        if default_label is None:
            return None
        return np.full(len(frame), default_label, dtype=np.int8)

    def _ensure_shared_crs(self, other: Points) -> tuple[Points, CRS | None]:
        """Convert another points object into the current CRS when needed."""
        if self.crs == other.crs:
            return other, self.crs

        if self.crs is None or other.crs is None:
            logger.warning(
                "Point CRS is missing. Assuming both point collections already "
                "share the same coordinate system."
            )
            return other, self.crs or other.crs

        return other.to_crs(self.crs), self.crs

    @property
    def values(self) -> np.ndarray:
        """Return the point coordinates with shape ``(n, 2)``."""
        return self._values

    @property
    def x(self) -> np.ndarray:
        """Return x coordinates."""
        return self._values[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Return y coordinates."""
        return self._values[:, 1]

    @property
    def labels(self) -> np.ndarray | None:
        """Return prompt labels."""
        return self._labels

    @property
    def has_label(self) -> bool:
        """Return whether labels are available."""
        return self._labels is not None

    @property
    def dtype(self) -> np.dtype:
        """Return the points dtype."""
        return self._values.dtype

    @property
    def crs(self) -> CRS | None:
        """Return the CRS."""
        return self._crs

    @property
    def center(self) -> tuple[float, float]:
        """Return the geometric center of all points."""
        return (float(np.mean(self.x)), float(np.mean(self.y)))

    @property
    def bounds(self) -> BoundingBox:
        """Return the minimal bounding box covering the points."""
        return BoundingBox(
            left=float(np.min(self.x)),
            bottom=float(np.min(self.y)),
            right=float(np.max(self.x)),
            top=float(np.max(self.y)),
            crs=self.crs,
        )

    def set_crs(self, crs: CrsLike) -> None:
        """Assign a CRS without reprojection."""
        self._crs = CRS.from_user_input(crs)

    def to_crs(self, crs: CrsLike) -> Points:
        """Reproject point coordinates."""
        if self.crs is None:
            msg = "Cannot reproject points without a source CRS."
            logger.error(msg)
            raise ValueError(msg)

        target_crs = CRS.from_user_input(crs)
        if self.crs == target_crs:
            return self

        transformer = Transformer.from_crs(self.crs, target_crs, always_xy=True)
        x, y = transformer.transform(self.x, self.y)
        return Points(np.column_stack([x, y]), labels=self.labels, crs=target_crs)

    def to_prompt(self) -> tuple[list[list[float]], list[int]]:
        """Return prompt-ready coordinates and labels."""
        if self.labels is None:
            msg = "Point labels are required to build a SAM prompt."
            logger.error(msg)
            raise ValueError(msg)
        return self.values.tolist(), self.labels.tolist()

    def to_sam_prompt(self) -> tuple[list[list[float]], list[int]]:
        """Alias for :meth:`to_prompt`."""
        return self.to_prompt()

    @classmethod
    def from_dataframe(
        cls,
        frame: gpd.GeoDataFrame | pd.DataFrame,
        x_field: str | None = None,
        y_field: str | None = None,
        label_field: str | None = None,
        default_label: PointPromptLabel | None = None,
        crs: CrsLike | None = None,
    ) -> Points:
        """Build points from a DataFrame or GeoDataFrame."""
        if isinstance(frame, gpd.GeoDataFrame) and frame.geometry is not None:
            points = np.column_stack(
                [frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy()]
            )
            frame_crs = frame.crs if frame.crs is not None else crs
        else:
            x_field, y_field = cls._ensure_fields(frame, x_field, y_field)
            points = frame[[x_field, y_field]].to_numpy()
            frame_crs = crs

        labels = cls._resolve_labels(frame, label_field, default_label)
        return cls(points, labels=labels, crs=frame_crs)

    @classmethod
    def from_file(
        cls,
        filename: PathLike,
        label_field: str | None = None,
        default_label: PointPromptLabel | None = None,
        **kwargs: object,
    ) -> Points:
        """Build points from a vector file."""
        frame = gpd.read_file(filename, **kwargs)
        exploded = frame.explode(index_parts=False, ignore_index=True)
        labels = cls._resolve_labels(exploded, label_field, default_label)
        points = np.column_stack(
            [exploded.geometry.x.to_numpy(), exploded.geometry.y.to_numpy()]
        )
        return cls(points, labels=labels, crs=exploded.crs)

    @classmethod
    def from_csv(
        cls,
        filename: PathLike,
        x_field: str | None = None,
        y_field: str | None = None,
        label_field: str | None = None,
        default_label: PointPromptLabel | None = None,
        crs: CrsLike | None = None,
        **kwargs: object,
    ) -> Points:
        """Build points from a CSV-like table."""
        frame = pd.read_csv(filename, **kwargs)
        return cls.from_dataframe(
            frame,
            x_field=x_field,
            y_field=y_field,
            label_field=label_field,
            default_label=default_label,
            crs=crs,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the points to a DataFrame."""
        frame = pd.DataFrame(self.values, columns=["x", "y"])
        if self.labels is not None:
            frame["labels"] = self.labels
        return frame

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert the points to a GeoDataFrame."""
        frame = self.to_dataframe()
        geometry = gpd.points_from_xy(frame["x"], frame["y"])
        return gpd.GeoDataFrame(frame, geometry=geometry, crs=self.crs)

    def to_file(self, filename: PathLike, **kwargs: object) -> None:
        """Write the points to a vector file."""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_geodataframe().to_file(output_path, **kwargs)
