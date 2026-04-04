"""Example script for testing cached SAM image features with multiple checkpoints."""
#%%
from __future__ import annotations

import logging
from pathlib import Path

from geosam import EncodedImageFeatures, SAMFeatureEncoder


#%%
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("/Users/fancy/Documents/SAM")
SOURCE_IMAGE = Path("/Users/fancy/Desktop/ultralytics/image.png")
CACHE_DIR = Path("artifacts/feature_cache")
POINTS = [[200, 150]]
LABELS = [1]
SKIPPED_PREFIXES = ("fastsam",)


def iter_supported_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Collect supported SAM and SAM2 checkpoints from a directory.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing local checkpoint files.

    Returns
    -------
    list[Path]
        Supported checkpoint paths sorted by filename.
    """
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    supported_checkpoints: list[Path] = []

    for checkpoint in checkpoints:
        if checkpoint.stem.lower().startswith(SKIPPED_PREFIXES):
            logger.info("Skipping unsupported checkpoint family: %s", checkpoint.name)
            continue
        supported_checkpoints.append(checkpoint)

    return supported_checkpoints


def run_checkpoint_trial(
    checkpoint_path: Path, source_image: Path, cache_dir: Path
) -> None:
    """Encode, save, load, and reuse features for one checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        SAM or SAM2 checkpoint path.
    source_image : Path
        Source image used for encoding and inference.
    cache_dir : Path
        Directory used to store cached feature files.
    """
    logger.info("Testing checkpoint: %s", checkpoint_path.name)
    encoder = SAMFeatureEncoder(checkpoint_path=checkpoint_path)

    encoded = encoder.encode(source_image)
    cache_path = cache_dir / f"{checkpoint_path.stem}.pt"
    encoded.save(cache_path)

    loaded = EncodedImageFeatures.load(cache_path)
    masks, boxes = encoder.inference_features(
        loaded,
        points=POINTS,
        labels=LABELS,
        multimask_output=True,
    )

    mask_count = 0 if masks is None else int(masks.shape[0])
    logger.info("Feature summary: %s", loaded.describe())
    logger.info("Saved feature file: %s", cache_path.resolve())
    logger.info(
        "Predicted %d mask(s) and %d box row(s).", mask_count, int(boxes.shape[0])
    )


def main() -> None:
    """Run the feature caching test across all supported local checkpoints.

    Raises
    ------
    FileNotFoundError
        If the checkpoint directory or source image is missing.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if not CHECKPOINT_DIR.exists():
        logger.error("Checkpoint directory does not exist: %s", CHECKPOINT_DIR)
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {CHECKPOINT_DIR}"
        )

    if not SOURCE_IMAGE.exists():
        logger.error("Source image does not exist: %s", SOURCE_IMAGE)
        raise FileNotFoundError(f"Source image does not exist: {SOURCE_IMAGE}")

    checkpoints = iter_supported_checkpoints(CHECKPOINT_DIR)
    if not checkpoints:
        logger.error("No supported checkpoints were found in %s", CHECKPOINT_DIR)
        raise FileNotFoundError(
            f"No supported checkpoints were found in {CHECKPOINT_DIR}"
        )

    for checkpoint_path in checkpoints:
        try:
            run_checkpoint_trial(checkpoint_path, SOURCE_IMAGE, CACHE_DIR)
        except Exception as error:  # noqa: BLE001
            logger.exception(
                "Checkpoint test failed for %s: %s", checkpoint_path.name, error
            )


if __name__ == "__main__":
    main()
