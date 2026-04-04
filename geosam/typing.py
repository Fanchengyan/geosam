"""Type aliases for geosam.

This module defines common type aliases used across the geosam package.
"""

from __future__ import annotations

from os import PathLike as OsPathLike
from typing import Any

from pyproj.crs import CRS
from typing_extensions import TypeAlias

CrsLike: TypeAlias = CRS | str | dict[str, Any] | int
PathLike: TypeAlias = str | OsPathLike
