"""Compute embeddings for images and patches."""

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import numpy as np
from tqdm import tqdm

from yolo_scout.core.constants import (
    DETECTION_FIELD,
    IMAGE_EMBEDDINGS_KEY,
    PATCH_EMBEDDINGS_KEY,
    get_field_name,
)
from yolo_scout.core.enums import DatasetTask
from yolo_scout.embeddings.preprocessing import extract_all_patch_crops
from yolo_scout.utils.logger import logger


def compute_embeddings(
    dataset: fo.Dataset,
    dataset_task: DatasetTask,
    model_kwargs: dict,
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
) -> dict[str, np.ndarray]:
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
    # Extract all crops
    all_crops, sample_id_per_crop = extract_all_patch_crops(
        dataset=dataset,
        patches_field=patches_field,
        dataset_task=dataset_task,
        background_color=(114, 114, 114),
        mask_background=mask_background,
    )

    if not all_crops:
        logger.warning("No crops extracted from dataset")
        return {}

    logger.info(f"Extracted {len(all_crops)} crops, computing embeddings...")

    # Batch inference
    all_embeddings_list = []

    for i in tqdm(range(0, len(all_crops), batch_size), desc="Computing embeddings"):
        batch = all_crops[i : i + batch_size]
        batch_embeds = model.embed_all(batch)

        # Convert to numpy array if needed
        if hasattr(batch_embeds, "cpu"):
            batch_embeds = batch_embeds.cpu().numpy()
        elif not isinstance(batch_embeds, np.ndarray):
            batch_embeds = np.array(batch_embeds)

        all_embeddings_list.append(batch_embeds)

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings_list)

    # Group embeddings by sample_id
    embeddings_dict = {}
    sample_id_per_crop_array = np.array(sample_id_per_crop)

    for sample_id in np.unique(sample_id_per_crop_array):
        # Find all embeddings belonging to this sample
        mask = sample_id_per_crop_array == sample_id
        embeddings_dict[sample_id] = all_embeddings[mask]

    logger.info(f"Successfully computed embeddings for {len(all_crops)} patches across {len(embeddings_dict)} samples")

    return embeddings_dict
