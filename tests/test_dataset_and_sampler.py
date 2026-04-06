from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from geosam.datasets import GridGeoSampler, RasterDataset, stack_samples
from geosam.query import BoundingBox


def _write_test_raster(path: Path) -> Path:
    data = np.arange(3 * 6 * 6, dtype=np.uint8).reshape(3, 6, 6)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=6,
        width=6,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0.0, 6.0, 1.0, 1.0),
    ) as dataset:
        dataset.write(data)
    return path


def test_raster_dataset_and_stack_samples(tmp_path: Path) -> None:
    raster_path = _write_test_raster(tmp_path / "sample.tif")
    dataset = RasterDataset(raster_path)
    query = BoundingBox(1.0, 1.0, 4.0, 4.0, crs="EPSG:4326")

    sample = dataset[query]
    batch = stack_samples([sample, sample])

    assert sample.shape == (3, 3)
    assert sample.image.shape == (3, 3, 3)
    assert batch["image"].shape == (2, 3, 3, 3)
    assert batch["bbox"][0] == sample.bbox


def test_grid_geo_sampler_is_deterministic(tmp_path: Path) -> None:
    raster_path = _write_test_raster(tmp_path / "sample.tif")
    dataset = RasterDataset(raster_path)
    sampler = GridGeoSampler(dataset, chip_size=4, overlap=2)

    chips = list(sampler)

    assert len(chips) == 4
    assert chips[0] == BoundingBox(0.0, 2.0, 4.0, 6.0, crs="EPSG:4326")
    assert chips[-1] == BoundingBox(2.0, 0.0, 6.0, 4.0, crs="EPSG:4326")


def test_raster_dataset_supports_output_resolution(tmp_path: Path) -> None:
    raster_path = _write_test_raster(tmp_path / "sample.tif")
    dataset = RasterDataset(raster_path, res=0.5)
    query = BoundingBox(1.0, 1.0, 4.0, 4.0, crs="EPSG:4326")

    sample = dataset[query]

    assert dataset.crs.to_string() == "EPSG:4326"
    assert sample.crs.to_string() == "EPSG:4326"
    assert sample.shape == (6, 6)


def test_raster_dataset_supports_output_crs(tmp_path: Path) -> None:
    raster_path = _write_test_raster(tmp_path / "sample.tif")
    dataset = RasterDataset(
        raster_path,
        crs="EPSG:3857",
        resampling=Resampling.average,
    )
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    left, bottom = transformer.transform(1.0, 1.0)
    right, top = transformer.transform(4.0, 4.0)
    query = BoundingBox(left, bottom, right, top, crs="EPSG:3857")

    sample = dataset[query]

    assert dataset.crs.to_string() == "EPSG:3857"
    assert sample.crs.to_string() == "EPSG:3857"
    assert sample.image.ndim == 3


def test_to_model_image_supports_pca_band_reduction() -> None:
    from geosam.datasets.raster import RasterSample

    chip = RasterSample(
        image=np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4),
        bbox=BoundingBox(0.0, 0.0, 1.0, 1.0, crs="EPSG:4326"),
        crs=rasterio.crs.CRS.from_epsg(4326),
        transform=from_origin(0.0, 1.0, 1.0, 1.0),
        shape=(4, 4),
        source_path="memory",
    )

    image_without_pca = chip.to_model_image()
    image_with_pca = chip.to_model_image(pca=True)

    assert image_without_pca.shape == (4, 4, 3)
    assert image_with_pca.shape == (4, 4, 3)
    assert image_without_pca.dtype == np.uint8
    assert image_with_pca.dtype == np.uint8
    assert not np.array_equal(image_without_pca, image_with_pca)
