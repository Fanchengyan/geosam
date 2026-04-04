"""A module for handling points query."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from collections.abc import Sequence as SequenceABC
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyproj.crs import CRS

from geosam.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from os import PathLike

    import numpy.typing as npt

    from geosam.typing import CrsLike


PointPromptLabel = Literal[0, 1]


class Points:
    """A class to represent a collection of points.

    with optional SAM point-prompt labels and coordinate reference system.
    """

    _values: np.ndarray
    _crs: CrsLike | None
    _labels: np.ndarray | None

    __slots__ = ["_crs", "_labels", "_values"]

    def __init__(
        self,
        points: Sequence[float | Sequence[float]] | npt.ArrayLike,
        labels: PointPromptLabel
        | Sequence[PointPromptLabel]
        | npt.ArrayLike
        | None = None,
        crs: CrsLike | None = None,
        dtype: npt.DTypeLike = np.float32,
        label_dtype: npt.DTypeLike = np.int8,
    ) -> None:
        """Initialize a Points object.

        Parameters
        ----------
        points : Sequence[float | Sequence[float]]
            The points to be sampled. The shape of the points can be (2) or (n, 2)
            where n is the number of points. The first column is the x coordinate
            and the second column is the y coordinate. if the shape is (2), the
            points will be reshaped to (1, 2).
        labels : {0, 1} or Sequence[{0, 1}] or ArrayLike, optional
            Point labels for SAM prompts. A scalar labels will be broadcast to all
            points. Label ``1`` means foreground and labels ``0`` means background.
        crs: CrsLike | None, optional, default: None
            The coordinate reference system of the points. Can be any object
            that can be passed to :meth:`pyproj.crs.CRS.from_user_input`.
        dtype : npt.DTypeLike, optional
            The data type of the points. Default is np.float32.
        label_dtype : npt.DTypeLike, optional
            The data type of the labels. Default is ``np.int8``.

        Raises
        ------
        ValueError
            If the shape of the points is not (n, 2).

        """
        self._values = np.asarray(points, dtype=dtype)
        self._crs = crs
        if self._values.ndim == 1:
            self._values = self._values.reshape(1, -1)
        if self._values.ndim != 2 or self._values.shape[1] != 2:
            msg = f"points must be a 2D array with 2 columns. Got {self._values}"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)
        self._labels = self._normalize_labels(labels, len(self), label_dtype)

    @staticmethod
    def _normalize_labels(
        labels: PointPromptLabel | Sequence[PointPromptLabel] | npt.ArrayLike | None,
        point_count: int,
        dtype: npt.DTypeLike,
    ) -> np.ndarray | None:
        """Normalize labels to a 1D NumPy array.

        Parameters
        ----------
        labels : {0, 1} or Sequence[{0, 1}] or ArrayLike or None
            Point labels to normalize.
        point_count : int
            Number of points.
        dtype : npt.DTypeLike
            Output data type for the labels.

        Returns
        -------
        np.ndarray | None
            Normalized labels array with shape ``(n,)``.

        Raises
        ------
        ValueError
            If labels are not one-dimensional, do not match the point count, or
            contain values other than ``0`` and ``1``.

        """
        if labels is None:
            return None

        labels = np.asarray(labels, dtype=dtype)
        if labels.ndim == 0:
            labels = np.full(point_count, labels.item(), dtype=dtype)
        elif labels.ndim != 1:
            msg = (
                "labels must be a scalar or 1D array whose length matches the "
                f"number of points. Got shape {labels.shape}."
            )
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)
        elif labels.shape[0] == 1 and point_count > 1:
            labels = np.full(point_count, labels[0], dtype=dtype)
        elif labels.shape[0] != point_count:
            msg = (
                "labels length must match the number of points. "
                f"Got {labels.shape[0]} labels for {point_count} points."
            )
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)

        if not np.isin(labels, [0, 1]).all():
            msg = (
                "labels values for SAM point prompts must be 0 or 1. "
                f"Got {labels.tolist()}."
            )
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)
        return labels

    def __len__(self) -> int:
        """Return the number of points."""
        return self._values.shape[0]

    def __iter__(self) -> Iterator[np.ndarray]:
        """Return an iterator of the points."""
        yield from self._values

    def __getitem__(self, key: int) -> Points:
        """Get the point by index."""
        labels = None if self.labels is None else self.labels[key]
        return Points(self._values[key, :], labels=labels, crs=self.crs)

    def __contains__(self, item: Points | Sequence[float]) -> bool:
        """Check if the item is in the points."""
        item_val: np.ndarray
        if isinstance(item, Points):
            item_val = item.values
        elif isinstance(item, (SequenceABC, np.ndarray)):
            item_val = np.array(item, dtype=np.float64)
        else:
            msg = f"item must be an Points or Sequence. Got {type(item)}"
            logger.error(msg)
            raise TypeError(msg)
        if item_val.ndim == 1:
            item_val = item_val.reshape(1, -1)
        if item_val.ndim != 2 or item_val.shape[1] != 2:
            msg = f"item must be a 2D array with shape (n, 2). Got {item_val.shape}."
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)

        matches = np.any(
            np.all(
                self._values[:, np.newaxis, :] == item_val[np.newaxis, :, :], axis=2
            ),
            axis=0,
        )
        return bool(np.all(matches))

    def __str__(self) -> str:
        """Return the string representation of the Points."""
        return (
            f"Points(count={len(self)}, crs='{self.crs}', has_label={self.has_label})"
        )

    def __repr__(self) -> str:
        """Return the string representation of the Points."""
        prefix = "Points:\n"
        middle = self.to_dataframe().to_string(max_rows=10)
        suffix = f"\n[count={len(self)}, crs='{self.crs}', has_label={self.has_label}]"

        return f"{prefix}{middle}{suffix}"

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:  # noqa: PLW3201
        """Return the values of the points as a numpy array."""
        if dtype is not None:
            return self._values.astype(dtype)
        return self._values

    def __add__(self, other: Points) -> Points:
        """Return the union of two Points."""
        if not isinstance(other, Points):
            msg = f"other must be an instance of Points. Got {type(other)}"
            logger.error(msg, stacklevel=2)
            raise TypeError(msg)

        other, crs_new = self._ensure_points_crs(other)
        if self.has_label != other.has_label:
            msg = (
                "Cannot combine labeled and unlabeled Points. "
                "Please align labels before merging."
            )
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)
        labels = None
        if self.has_label and other.has_label:
            labels = np.concatenate([self.labels, other.labels])

        return Points(
            np.vstack([self.values, other.values]), labels=labels, crs=crs_new
        )

    def __sub__(self, other: Points) -> Points | None:
        """Return the set difference of two Points."""
        if not isinstance(other, Points):
            msg = f"other must be an instance of Points. Got {type(other)}"
            logger.error(msg, stacklevel=2)
            raise TypeError(msg)

        other, crs_new = self._ensure_points_crs(other)

        mask = ~np.any(
            np.all(
                self._values[:, np.newaxis, :] == other._values[np.newaxis, :, :],
                axis=2,
            ),
            axis=1,
        )
        values = self._values[mask]
        if len(values) == 0:
            return None
        labels = None if self.labels is None else self.labels[mask]
        return Points(values, labels=labels, crs=crs_new)

    def _ensure_points_crs(self, other: Points) -> tuple[Points, CrsLike | None]:
        """Ensure the coordinate reference system of the points are the same."""
        if self.crs != other.crs:
            if self.crs is None or other.crs is None:
                crs_new = self.crs or other.crs
                logger.warning(
                    "Cannot find the coordinate reference system of the points. "
                    "The crs of two points will assume to be the same."
                )
            else:
                other = other.to_crs(self.crs)
                crs_new = self.crs
        else:
            crs_new = self.crs
        return other, crs_new

    @staticmethod
    def _find_field(
        gdf: gpd.GeoDataFrame | pd.DataFrame, field_names: list[str]
    ) -> str | None:
        """Find the field name in the GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame to be searched.
        field_names : list[str]
            The field names to be searched.

        Returns
        -------
        str | None
            The field name found. If not found, return None.

        """
        for name in gdf.columns:
            if name.lower() in field_names:
                return name
        return None

    @classmethod
    def _ensure_fields(
        cls,
        gdf: gpd.GeoDataFrame | pd.DataFrame,
        x_field: str | None,
        y_field: str | None,
    ) -> tuple[str, str]:
        """Parse the field from the GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame | pd.DataFrame
            The GeoDataFrame to be parsed.
        x_field : str
            The field name of the x coordinate.
        y_field : str
            The field name of the y coordinate.

        Returns
        -------
        tuple[str, str]
            The field names of the x and y coordinates.

        Raises
        ------
        ValueError
            If the field does not specify and cannot be found automatically.

        """
        if x_field is None:
            x_field = cls._find_field(
                gdf,
                ["x", "xs", "lon", "longitude", "long", "longs", "longitudes"],
            )
            if x_field is None:
                msg = (
                    "Cannot find the field name of the x coordinate. "
                    "Please provide the field name manually."
                )
                logger.error(msg, stacklevel=2)
                raise ValueError(msg)
        if y_field is None:
            y_field = cls._find_field(
                gdf,
                ["y", "ys", "lat", "latitude", "lats", "latitudes"],
            )
            if y_field is None:
                msg = (
                    "Cannot find the field name of the y coordinate. "
                    "Please provide the field name manually."
                )
                logger.error(msg, stacklevel=2)
                raise ValueError(msg)
        return x_field, y_field

    @property
    def x(self) -> np.ndarray:
        """Return the x coordinates of the points."""
        return self._values[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Return the y coordinates of the points."""
        return self._values[:, 1]

    @property
    def values(self) -> np.ndarray:
        """Return the values of the points with shape (n, 2).

        n is the number of points.
        """
        return self._values

    @property
    def labels(self) -> np.ndarray | None:
        """Return the point labels used by SAM prompts."""
        return self._labels

    @property
    def has_label(self) -> bool:
        """Return whether the points include SAM labels."""
        return self._labels is not None

    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the points."""
        return self._values.dtype

    @property
    def crs(self) -> CrsLike | None:
        """Return the coordinate reference system of the points."""
        return self._crs

    def set_crs(self, crs: CrsLike) -> None:
        """Set the coordinate reference system of the points.

        .. warning::
            This method will only set the crs attribute without converting the
            points to a new coordinate reference system. If you want to convert
            the points values to a new coordinate, please use :meth:`to_crs`
        """
        self._crs = CRS.from_user_input(crs)

    def to_crs(self, crs: CrsLike) -> Points:
        """Convert the points values to a new coordinate reference system.

        Parameters
        ----------
        crs : CrsLike
            The new coordinate reference system. Can be any object that can be
            passed to :meth:`pyproj.crs.CRS.from_user_input`.

        Returns
        -------
        Points
            The points in the new coordinate reference system.

        """
        if self.crs is None:
            msg = (
                "The current coordinate reference system is None. "
                "Please set the crs using set_crs() first."
            )
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)
        crs = CRS.from_user_input(crs)

        if self.crs == crs:
            return self

        transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
        x, y = transformer.transform(self.x, self.y)
        values = np.column_stack((x, y))
        return Points(values, labels=self.labels, crs=crs)

    def to_prompt(self) -> tuple[list[list[float]], list[int]]:
        """Convert points to SAM point-prompt arrays.

        Returns
        -------
        tuple[list[list[float]], list[int]]
            A tuple of ``(points, labels)`` for SAM prompt inference.

        Raises
        ------
        ValueError
            If labels are not available.

        """
        if self.labels is None:
            msg = "Point labels are required to build a SAM point prompt."
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)
        return self.values.tolist(), self.labels.tolist()

    @classmethod
    def _resolve_labels(
        cls,
        gdf: gpd.GeoDataFrame | pd.DataFrame,
        label_field: str | None,
        default_label: PointPromptLabel | None,
    ) -> np.ndarray | None:
        """Resolve labels from a tabular dataset.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame | pd.DataFrame
            Tabular source containing point data.
        label_field : str | None
            Name of the labels field. If ``None``, try to detect it automatically.
        default_label : {0, 1} | None
            Fallback labels used when no labels field is available.

        Returns
        -------
        np.ndarray | None
            Resolved labels array.

        Raises
        ------
        ValueError
            If ``label_field`` is specified but not found.

        """
        if label_field is None:
            label_field = cls._find_field(
                gdf,
                [
                    "labels",
                    "labels",
                    "point_label",
                    "point_labels",
                    "prompt_label",
                    "prompt_labels",
                ],
            )

        if label_field is not None:
            if label_field not in gdf.columns:
                msg = f"Cannot find labels field '{label_field}' in the input data."
                logger.error(msg, stacklevel=2)
                raise ValueError(msg)
            return gdf[label_field].to_numpy()

        if default_label is None:
            return None
        return np.full(len(gdf), default_label, dtype=np.int8)

    @classmethod
    def from_dataframe(
        cls,
        gdf: gpd.GeoDataFrame | pd.DataFrame,
        x_field: str | None = None,
        y_field: str | None = None,
        label_field: str | None = None,
        default_label: PointPromptLabel | None = None,
        crs: CrsLike | None = None,
    ) -> Points:
        """Initialize a Points object from a GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame | pd.DataFrame
            The GeoDataFrame to be parsed.
        x_field, y_field : str, optional, default: None
            The field name of the x/y coordinates if geometry does not exist.
            If None, will try to find the field name automatically from
            following fields (case insensitive):

            - **x**: x, xs, lon, longitude, long, longs, longitudes
            - **y**: y, ys, lat, latitude, lats, latitudes
        label_field : str | None, optional, default: None
            Field name storing SAM point labels. If ``None``, the method will
            try to detect common labels field names automatically.
        default_label : {0, 1} | None, optional, default: None
            Default SAM point labels used when no labels field is available.
        crs : CrsLike | None, optional, default: None
            The coordinate reference system of the points. Can be any object
            that can be passed to :meth:`pyproj.crs.CRS.from_user_input`. If
            gdf has a valid crs, it will be ignored.

        Returns
        -------
        Points
            The Points object.

        """
        if isinstance(gdf, gpd.GeoDataFrame) and hasattr(gdf, "geometry"):
            points = list(zip(gdf.geometry.x, gdf.geometry.y))
            crs = gdf.crs if gdf.crs is not None else crs
        else:
            x_field, y_field = cls._ensure_fields(gdf, x_field, y_field)
            points = gdf[[x_field, y_field]].values.tolist()
        labels = cls._resolve_labels(gdf, label_field, default_label)

        return cls(points, labels=labels, crs=crs)

    @classmethod
    def from_file(
        cls,
        filename: PathLike,
        label_field: str | None = None,
        default_label: PointPromptLabel | None = None,
        **kwargs,
    ) -> Points:
        """Initialize a Points object from a file.

        Parameters
        ----------
        filename : PathLike
            The path to the shapefile. file type can be any type that can be
            passed to :func:`geopandas.read_file`.
        label_field : str | None, optional, default: None
            Field name storing SAM point labels.
        default_label : {0, 1} | None, optional, default: None
            Default SAM point labels used when no labels field is available.
        **kwargs : dict
            Other parameters passed to :func:`geopandas.read_file`.

        Returns
        -------
        Points
            The Points object.

        """
        gdf = gpd.read_file(filename, **kwargs)
        exploded_gdf = gdf.explode(index_parts=False, ignore_index=True)
        geometry = exploded_gdf.geometry

        points = list(zip(geometry.x, geometry.y))
        labels = cls._resolve_labels(exploded_gdf, label_field, default_label)
        return cls(points, labels=labels, crs=exploded_gdf.crs)

    @classmethod
    def from_csv(
        cls,
        filename: PathLike,
        x_field: str | None = None,
        y_field: str | None = None,
        label_field: str | None = None,
        default_label: PointPromptLabel | None = None,
        crs: CrsLike | None = None,
        **kwargs,
    ) -> Points:
        """Initialize a Points object from a csv/txt file.

        Parameters
        ----------
        filename : PathLike
            The path to the csv/txt file.
        x_field, y_field : str, optional, default: "auto"
            The field name of the x/y coordinates. If "auto", will try to
            find the field name automatically from following fields (case insensitive):

            * ``x`` : x, xs, lon, longitude
            * ``y`` : y, ys, lat, latitude
        label_field : str | None, optional, default: None
            Field name storing SAM point labels.
        default_label : {0, 1} | None, optional, default: None
            Default SAM point labels used when no labels field is available.
        crs : CrsLike | None, optional, default: None
            The coordinate reference system of the points. Can be any object
            that can be passed to :meth:`pyproj.crs.CRS.from_user_input`.
        **kwargs : dict
            Other parameters passed to :func:`pandas.read_csv`.

        Returns
        -------
        Points
            The Points object.

        """
        df = pd.read_csv(filename, **kwargs)
        return cls.from_dataframe(
            df,
            x_field=x_field,
            y_field=y_field,
            label_field=label_field,
            default_label=default_label,
            crs=crs,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the Points to a DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame with columns ``x`` and ``y``. If labels are available,
            a ``labels`` column is also included.

        """
        df = pd.DataFrame(self._values, columns=["x", "y"])
        if self.labels is not None:
            df["labels"] = self.labels
        return df

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert the Points to a GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            The GeoDataFrame.

        """
        df = self.to_dataframe()
        geometry = gpd.points_from_xy(df["x"], df["y"])
        return gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)

    def to_file(self, filename: PathLike, **kwargs) -> None:
        """Save the Points to a file.

        Parameters
        ----------
        filename : PathLike
            The path to the file.
        **kwargs : dict
            Other parameters passed to :meth:`geopandas.GeoDataFrame.to_file`.

        """
        self.to_geodataframe().to_file(filename, **kwargs)
