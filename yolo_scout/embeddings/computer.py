"""Compute embeddings for images and patches."""

from collections import defaultdict
from typing import Dict, List

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import numpy as np
from PIL import Image
from tqdm import tqdm

from yolo_scout.core.constants import (
    DETECTION_FIELD,
    IMAGE_EMBEDDINGS_KEY,
    PATCH_EMBEDDINGS_KEY,
    get_field_name,
)
from yolo_scout.core.enums import DatasetTask
from yolo_scout.embeddings.preprocessing import iter_patch_crops
from yolo_scout.utils.logger import logger


def compute_embeddings(
    dataset: fo.Dataset,
    dataset_task: DatasetTask,
    model_kwargs: Dict,
    batch_size: int,
    mask_background: bool = True,
) -> None:
    """
    Compute embeddings and visualizations for the dataset.

    Args:
        dataset: FiftyOne dataset
        dataset_task: Dataset task type
        model_kwargs: Model configuration kwargs
        batch_size: Batch size for processing
        mask_background: Whether to mask background in patch crops for segment/obb tasks
    """
    # Load embeddings model
    try:
        model = foz.load_zoo_model("open-clip-torch", **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to load embeddings model: {e}")
        raise

    # Compute image embeddings
    logger.info("Computing image embeddings and visualization...")
    try:
        fob.compute_visualization(
            dataset,
            model=model,
            method="umap",
            brain_key=IMAGE_EMBEDDINGS_KEY,
            batch_size=batch_size,
            seed=0,
        )
        logger.info("Image embeddings and visualization computed successfully")
    except Exception as e:
        logger.error(f"Failed to compute image embeddings: {e}")
        raise

    # Compute patch embeddings if not classification
    if dataset_task != DatasetTask.CLASSIFICATION:
        logger.info("\nComputing patch embeddings and visualization...")

        # Determine patches field
        patches_field = get_field_name(task=dataset_task)

        # For pose, we use bounding boxes for patches
        if dataset_task == DatasetTask.POSE:
            patches_field = DETECTION_FIELD

        try:
            # Compute embeddings with optional background masking for segmentation/OBB
            patch_embeddings = _compute_patch_embeddings(
                dataset=dataset,
                patches_field=patches_field,
                model=model,
                dataset_task=dataset_task,
                batch_size=batch_size,
                mask_background=mask_background,
            )

            # Pass pre-computed embeddings to FiftyOne
            fob.compute_visualization(
                dataset,
                patches_field=patches_field,
                embeddings=patch_embeddings,
                method="umap",
                brain_key=PATCH_EMBEDDINGS_KEY,
                seed=0,
            )

            logger.info("Patch embeddings and visualization computed successfully")

        except Exception as e:
            logger.error(f"Failed to compute patch embeddings: {e}")
            raise


def _compute_patch_embeddings(
    dataset: fo.Dataset,
    patches_field: str,
    model,
    dataset_task: DatasetTask,
    batch_size: int,
    mask_background: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for all patches with optional background masking.

    Args:
        dataset: FiftyOne dataset
        patches_field: Field name containing patches
        model: Model with embed_all() method
        dataset_task: Dataset task type
        batch_size: Batch size for model inference
        mask_background: Whether to mask background for segment/obb tasks

    Returns:
        Dict mapping sample_id -> (num_patches, embedding_dim) numpy array
    """
    per_sample_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    crop_buffer: List[Image.Image] = []
    sample_id_buffer: List[str] = []
    total_crops = 0

    def _embed_buffer() -> None:
        batch_embeds = model.embed_all(crop_buffer)

        # Convert to numpy array if needed
        if hasattr(batch_embeds, "cpu"):
            batch_embeds = batch_embeds.cpu().numpy()
        elif not isinstance(batch_embeds, np.ndarray):
            batch_embeds = np.array(batch_embeds)

        sample_id_buffer_array = np.array(sample_id_buffer)
        for sample_id in np.unique(sample_id_buffer_array):
            mask = sample_id_buffer_array == sample_id
            per_sample_embeddings[sample_id].append(batch_embeds[mask])

    # Stream crops and embed them in bounded-size batches, so at most
    # `batch_size` crops are held in memory at once regardless of dataset size
    crop_stream = iter_patch_crops(
        dataset=dataset,
        patches_field=patches_field,
        dataset_task=dataset_task,
        background_color=(114, 114, 114),
        mask_background=mask_background,
    )

    for sample_id, crops in tqdm(crop_stream, desc="Computing embeddings"):
        total_crops += len(crops)
        crop_buffer.extend(crops)
        sample_id_buffer.extend([sample_id] * len(crops))

        while len(crop_buffer) >= batch_size:
            crop_buffer, remainder = crop_buffer[:batch_size], crop_buffer[batch_size:]
            sample_id_buffer, sample_id_remainder = sample_id_buffer[:batch_size], sample_id_buffer[batch_size:]
            _embed_buffer()
            crop_buffer, sample_id_buffer = remainder, sample_id_remainder

    if crop_buffer:
        _embed_buffer()

    if total_crops == 0:
        logger.warning("No crops extracted from dataset")
        return {}

    # Concatenate each sample's accumulated batch-chunks into a single array
    embeddings_dict = {sample_id: np.vstack(chunks) for sample_id, chunks in per_sample_embeddings.items()}

    logger.info(f"Successfully computed embeddings for {total_crops} patches across {len(embeddings_dict)} samples")

    return embeddings_dict
