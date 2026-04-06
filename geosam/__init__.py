"""Public package interface for :mod:`geosam`."""

from geosam.datasets import (
    GeoGrid,
    GridGeoSampler,
    RasterDataset,
    RasterSample,
    stack_samples,
)
from geosam.engines import (
    FeatureCacheBuilder,
    FeatureQueryEngine,
    OnlineQueryCache,
    OnlineQueryEngine,
    QueryResult,
)
from geosam.feature_encoder import SAMFeatureEncoder
from geosam.models import EncodedImageFeatures, ModelSpec, build_model_adapter
from geosam.query import (
    BoundingBox,
    Points,
    PromptSet,
    bbox_to_prompt,
    points_to_prompt,
    query_bounds,
    query_center,
    window_from_center,
)
from geosam.runtime import (
    DEFAULT_MODEL_REPOSITORY,
    MODEL_DEFINITIONS,
    FeatureSourceSummary,
    ModelDefinition,
    chip_extent_rectangles_for_source,
    create_model_spec,
    create_model_spec_from_checkpoint,
    dependency_status,
    describe_feature_source,
    get_model_definition,
    get_model_display_items,
    infer_model_id_from_checkpoint_path,
    resolve_feature_manifest_path,
)
from geosam.vectorization import MaskVectorizer

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
    "create_model_spec",
    "create_model_spec_from_checkpoint",
    "dependency_status",
    "describe_feature_source",
    "get_model_definition",
    "get_model_display_items",
    "infer_model_id_from_checkpoint_path",
    "points_to_prompt",
    "query_bounds",
    "query_center",
    "resolve_feature_manifest_path",
    "stack_samples",
    "window_from_center",
]
