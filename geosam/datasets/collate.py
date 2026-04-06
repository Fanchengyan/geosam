"""Batch collation helpers for raster samples."""

from __future__ import annotations

from typing import Any

import numpy as np

from geosam.datasets.raster import RasterSample
from geosam.logging import setup_logger

logger = setup_logger(__name__)


def stack_samples(samples: list[RasterSample | dict[str, Any]]) -> dict[str, Any]:
    """Stack raster samples into a single batch dictionary."""
    if len(samples) == 0:
        msg = "stack_samples requires at least one sample."
        logger.error(msg)
        raise ValueError(msg)

    normalized: list[dict[str, Any]] = []
    for sample in samples:
        if isinstance(sample, RasterSample):
            normalized.append(sample.to_dict())
        else:
            normalized.append(sample)

    images = [entry["image"] for entry in normalized]
    if not all(image.shape == images[0].shape for image in images):
        msg = "All sample images must share the same shape for stacking."
        logger.error(msg)
        raise ValueError(msg)

    return {
        "image": np.stack(images, axis=0),
        "bbox": [entry["bbox"] for entry in normalized],
        "crs": [entry["crs"] for entry in normalized],
        "transform": [entry["transform"] for entry in normalized],
        "shape": [entry["shape"] for entry in normalized],
        "source_path": [entry["source_path"] for entry in normalized],
    }
