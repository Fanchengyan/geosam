"""Public package interface for :mod:`geosam`.

Public objects are loaded lazily so lightweight runtime configuration modules
can be imported in QGIS before native geospatial dependencies such as
``rasterio`` or ``pyproj`` are imported.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "DEFAULT_MODEL_REPOSITORY": "geosam.runtime",
    "MODEL_DEFINITIONS": "geosam.runtime",
    "BoundingBox": "geosam.query",
    "EncodedImageFeatures": "geosam.models",
    "FeatureCacheBuilder": "geosam.engines",
    "FeatureQueryEngine": "geosam.engines",
    "FeatureSourceSummary": "geosam.runtime",
    "GeoGrid": "geosam.datasets",
    "GridGeoSampler": "geosam.datasets",
    "MaskVectorizer": "geosam.vectorization",
    "ModelDefinition": "geosam.runtime",
    "ModelSpec": "geosam.models",
    "OnlineQueryCache": "geosam.engines",
    "OnlineQueryEngine": "geosam.engines",
    "Points": "geosam.query",
    "PromptSet": "geosam.query",
    "QueryResult": "geosam.engines",
    "RasterDataset": "geosam.datasets",
    "RasterSample": "geosam.datasets",
    "SAMFeatureEncoder": "geosam.feature_encoder",
    "bbox_to_prompt": "geosam.query",
    "build_model_adapter": "geosam.models",
    "chip_extent_rectangles_for_source": "geosam.runtime",
    "configure_runtime": "geosam.context",
    "create_model_spec": "geosam.runtime",
    "create_model_spec_from_checkpoint": "geosam.runtime",
    "dependency_status": "geosam.runtime",
    "describe_feature_source": "geosam.runtime",
    "get_model_definition": "geosam.runtime",
    "get_model_display_items": "geosam.runtime",
    "get_runtime": "geosam.context",
    "infer_model_id_from_checkpoint_path": "geosam.runtime",
    "points_to_prompt": "geosam.query",
    "query_bounds": "geosam.query",
    "query_center": "geosam.query",
    "resolve_feature_manifest_path": "geosam.runtime",
    "runtime_context": "geosam.context",
    "stack_samples": "geosam.datasets",
    "window_from_center": "geosam.query",
}

__all__ = [
    "DEFAULT_MODEL_REPOSITORY",
    "MODEL_DEFINITIONS",
    "BoundingBox",
    "EncodedImageFeatures",
    "FeatureCacheBuilder",
    "FeatureQueryEngine",
    "FeatureSourceSummary",
    "GeoGrid",
    "GridGeoSampler",
    "MaskVectorizer",
    "ModelDefinition",
    "ModelSpec",
    "OnlineQueryCache",
    "OnlineQueryEngine",
    "Points",
    "PromptSet",
    "QueryResult",
    "RasterDataset",
    "RasterSample",
    "SAMFeatureEncoder",
    "bbox_to_prompt",
    "build_model_adapter",
    "chip_extent_rectangles_for_source",
    "configure_runtime",
    "create_model_spec",
    "create_model_spec_from_checkpoint",
    "dependency_status",
    "describe_feature_source",
    "get_model_definition",
    "get_model_display_items",
    "get_runtime",
    "infer_model_id_from_checkpoint_path",
    "points_to_prompt",
    "query_bounds",
    "query_center",
    "resolve_feature_manifest_path",
    "runtime_context",
    "stack_samples",
    "window_from_center",
]


def __getattr__(name: str) -> Any:
    """Load public exports on first access.

    Parameters
    ----------
    name : str
        Public export name.

    Returns
    -------
    Any
        Exported object.

    Raises
    ------
    AttributeError
        If ``name`` is not a public export.

    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        message = f"module 'geosam' has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return public module attributes."""
    return sorted([*globals(), *__all__])
