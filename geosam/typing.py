"""Shared type aliases for :mod:`geosam`.

This module centralizes reusable type aliases shared across the package.
"""

from __future__ import annotations

from os import PathLike as OsPathLike
from typing import Any, Union

from pyproj.crs import CRS
from typing_extensions import TypeAlias

CrsLike: TypeAlias = Union[CRS, str, dict[str, Any], int]
PathLike: TypeAlias = Union[str, OsPathLike[str]]
Shape2D: TypeAlias = tuple[int, int]
PixelCoordinate: TypeAlias = tuple[float, float]
