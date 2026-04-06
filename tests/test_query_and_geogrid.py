from __future__ import annotations

import numpy as np
from rasterio import Affine

from geosam.datasets import GeoGrid
from geosam.query import (
    BoundingBox,
    Points,
    bbox_to_prompt,
    points_to_prompt,
)


def test_points_and_bbox_prompt_conversion() -> None:
    grid = GeoGrid(
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 8.0),
        shape=(8, 8),
        crs="EPSG:4326",
    )
    points = Points([[1.5, 6.5], [3.5, 4.5]], labels=[1, 0], crs="EPSG:4326")
    bbox = BoundingBox(1.0, 4.0, 4.0, 7.0, crs="EPSG:4326")

    point_prompt, point_labels = points_to_prompt(points, grid)
    bbox_prompt = bbox_to_prompt(bbox, grid)

    np.testing.assert_allclose(point_prompt, np.array([[1.5, 1.5], [3.5, 3.5]]))
    np.testing.assert_array_equal(point_labels, np.array([1, 0], dtype=np.int8))
    assert bbox_prompt == (1.0, 1.0, 4.0, 4.0)


def test_geogrid_window_and_scaled_prompts() -> None:
    grid = GeoGrid(
        transform=Affine(2.0, 0.0, 100.0, 0.0, -2.0, 200.0),
        shape=(10, 10),
        crs="EPSG:3857",
    )
    window = grid.window(row_off=2, col_off=3, height=4, width=4)
    points = Points([[107.0, 193.0]], labels=[1], crs="EPSG:3857")

    prompt, _ = points_to_prompt(points, window, dst_shape=(8, 8))

    np.testing.assert_allclose(prompt, np.array([[1.0, 3.0]], dtype=np.float32))
    assert window.bounds == BoundingBox(106.0, 188.0, 114.0, 196.0, crs="EPSG:3857")
