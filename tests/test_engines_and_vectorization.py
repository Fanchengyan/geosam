from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from geosam.datasets import RasterDataset
from geosam.engines import (
    FeatureCacheBuilder,
    FeatureQueryEngine,
    OnlineQueryEngine,
    QueryResult,
)
from geosam.models import EncodedImageFeatures, GeoSamPrediction, ModelSpec
from geosam.query import BoundingBox, Points, PromptSet
from geosam.vectorization import MaskVectorizer


class DummyAdapter:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.last_predict_image_kwargs: dict[str, object] | None = None
        self.last_predict_features_kwargs: dict[str, object] | None = None

    def encode_image(self, image: np.ndarray) -> EncodedImageFeatures:
        height, width = image.shape[:2]
        return EncodedImageFeatures(
            model_type=self.spec.model_type,
            checkpoint_path=self.spec.resolved_checkpoint_path,
            src_shape=(height, width),
            dst_shape=(height, width),
            features=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        )

    def predict_image(self, image: np.ndarray, **kwargs: object) -> GeoSamPrediction:
        self.last_predict_image_kwargs = kwargs
        return self.predict_features(self.encode_image(image))

    def predict_features(
        self,
        encoded: EncodedImageFeatures,
        **kwargs: object,
    ) -> GeoSamPrediction:
        self.last_predict_features_kwargs = kwargs
        height, width = encoded.src_shape
        masks = torch.ones((1, height, width), dtype=torch.bool)
        boxes = torch.tensor([[0.0, 0.0, width, height, 0.95, 0.0]])
        return GeoSamPrediction(masks=masks, boxes=boxes)


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


def _write_manifest_pickle(frame: gpd.GeoDataFrame, manifest_path: Path) -> Path:
    frame.to_pickle(manifest_path)
    return manifest_path


def _read_manifest_pickle(manifest_path: Path) -> gpd.GeoDataFrame:
    return pd.read_pickle(manifest_path).set_crs("EPSG:4326", allow_override=True)


def test_online_query_engine_and_missing_crs(tmp_path: Path, monkeypatch) -> None:
    raster_path = _write_test_raster(tmp_path / "sample.tif")
    dataset = RasterDataset(raster_path)
    model_spec = ModelSpec("sam", checkpoint_path="dummy.pt")
    adapter = DummyAdapter(model_spec)
    monkeypatch.setattr("geosam.engines.build_model_adapter", lambda _spec: adapter)

    engine = OnlineQueryEngine(dataset, model_spec)
    query = Points([[1.5, 4.5]], labels=[1], crs="EPSG:4326")
    result = engine.query(query)

    assert result.mask_array.shape == (1, 6, 6)
    np.testing.assert_allclose(result.scores, np.array([0.95], dtype=np.float32))

    no_crs_query = BoundingBox(1.0, 1.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="CRS"):
        engine.query(no_crs_query)


def test_online_query_engine_supports_composite_prompts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raster_path = _write_test_raster(tmp_path / "sample.tif")
    dataset = RasterDataset(raster_path)
    model_spec = ModelSpec("sam", checkpoint_path="dummy.pt")
    adapter = DummyAdapter(model_spec)
    monkeypatch.setattr("geosam.engines.build_model_adapter", lambda _spec: adapter)

    engine = OnlineQueryEngine(dataset, model_spec)
    query = PromptSet(
        points=Points([[1.5, 4.5]], labels=[1], crs="EPSG:4326"),
        bbox=BoundingBox(1.0, 4.0, 2.0, 5.0, crs="EPSG:4326"),
    )

    result = engine.query(query)

    assert result.mask_array.shape == (1, 6, 6)
    assert adapter.last_predict_image_kwargs is not None
    assert "points" in adapter.last_predict_image_kwargs
    assert "bboxes" in adapter.last_predict_image_kwargs


def test_feature_cache_and_feature_query_engine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raster_path = _write_test_raster(tmp_path / "sample.tif")
    dataset = RasterDataset(raster_path)
    model_spec = ModelSpec(
        "sam",
        checkpoint_path="dummy.pt",
        supports_feature_reuse=True,
    )
    adapter = DummyAdapter(model_spec)
    monkeypatch.setattr("geosam.engines.build_model_adapter", lambda _spec: adapter)
    monkeypatch.setattr(
        FeatureCacheBuilder,
        "write_manifest",
        staticmethod(_write_manifest_pickle),
    )
    monkeypatch.setattr(
        FeatureQueryEngine,
        "load_manifest",
        staticmethod(_read_manifest_pickle),
    )

    builder = FeatureCacheBuilder(
        dataset,
        model_spec,
        tmp_path / "cache",
        chip_size=4,
        overlap=2,
    )
    manifest_path = builder.build(tmp_path / "cache" / "manifest.pkl")

    engine = FeatureQueryEngine(manifest_path, model_spec)
    query = BoundingBox(0.5, 2.5, 1.5, 3.5, crs="EPSG:4326")
    result = engine.query(query)

    assert result.chip_id == "chip_000000"
    assert result.mask_array.shape[0] == 1
    assert Path(engine.manifest.iloc[0]["feature_path"]).exists()
    assert FeatureQueryEngine.resolve_manifest_path(tmp_path / "cache") == manifest_path


def test_feature_query_engine_supports_composite_prompts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raster_path = _write_test_raster(tmp_path / "sample.tif")
    dataset = RasterDataset(raster_path)
    model_spec = ModelSpec(
        "sam",
        checkpoint_path="dummy.pt",
        supports_feature_reuse=True,
    )
    adapter = DummyAdapter(model_spec)
    monkeypatch.setattr("geosam.engines.build_model_adapter", lambda _spec: adapter)
    monkeypatch.setattr(
        FeatureCacheBuilder,
        "write_manifest",
        staticmethod(_write_manifest_pickle),
    )
    monkeypatch.setattr(
        FeatureQueryEngine,
        "load_manifest",
        staticmethod(_read_manifest_pickle),
    )

    builder = FeatureCacheBuilder(
        dataset,
        model_spec,
        tmp_path / "cache",
        chip_size=4,
        overlap=2,
    )
    manifest_path = builder.build(tmp_path / "cache" / "manifest.pkl")

    engine = FeatureQueryEngine(manifest_path.parent, model_spec)
    query = PromptSet(
        points=Points([[1.5, 4.5]], labels=[1], crs="EPSG:4326"),
        bbox=BoundingBox(0.5, 3.5, 2.0, 5.0, crs="EPSG:4326"),
    )

    result = engine.query(query)

    assert result.chip_id == "chip_000000"
    assert adapter.last_predict_features_kwargs is not None
    assert "points" in adapter.last_predict_features_kwargs
    assert "bboxes" in adapter.last_predict_features_kwargs


def test_mask_vectorizer_exports_geojson(tmp_path: Path) -> None:
    mask = np.zeros((1, 4, 4), dtype=bool)
    mask[0, 1:3, 1:3] = True
    vectorizer = MaskVectorizer(
        mask,
        transform=from_origin(0.0, 4.0, 1.0, 1.0),
        crs="EPSG:4326",
    )

    frame = vectorizer.to_geodataframe(properties={"score": 0.9})
    payload = vectorizer.to_geojson(
        properties={"score": 0.9},
    )
    output_path = vectorizer.write_geojson(
        tmp_path / "mask.geojson",
        properties={"score": 0.9},
    )

    assert len(frame) == 1
    assert payload["features"][0]["properties"]["score"] == 0.9
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert file_payload["features"][0]["properties"]["score"] == 0.9


def test_mask_vectorizer_from_query_result(tmp_path: Path) -> None:
    query_result = QueryResult(
        mask_array=np.ones((1, 4, 4), dtype=bool),
        mask_transform=from_origin(0.0, 4.0, 1.0, 1.0),
        mask_crs=rasterio.crs.CRS.from_epsg(4326),
        query_bounds=BoundingBox(0.0, 0.0, 4.0, 4.0, crs="EPSG:4326"),
        chip_bounds=BoundingBox(0.0, 0.0, 4.0, 4.0, crs="EPSG:4326"),
        scores=np.array([0.95], dtype=np.float32),
        source_path="sample.tif",
        chip_id="chip_000001",
        model_type="sam2",
    )

    vectorizer = MaskVectorizer.from_query_result(query_result)
    frame = vectorizer.to_geodataframe(properties={"score": 0.95})
    payload = vectorizer.to_geojson(
        properties={"score": 0.95},
    )
    output_path = vectorizer.write_geojson(
        tmp_path / "query_result_mask.geojson",
        properties={"score": 0.95},
    )

    assert len(frame) == 1
    assert frame.iloc[0]["source_path"] == "sample.tif"
    assert payload["features"][0]["properties"]["chip_id"] == "chip_000001"
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert file_payload["features"][0]["properties"]["chip_id"] == "chip_000001"
