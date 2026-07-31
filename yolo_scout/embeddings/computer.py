"""Compute embeddings for images and patches."""

import os

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import numpy as np
import torch
import umap
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

# Caps FiftyOne's per-worker mongod connections to avoid overwhelming it.
MAX_IMAGE_EMBEDDING_WORKERS = 8

# Keeps PyTorch's own thread pool from competing with the crop-extraction pool.
CPU_INTRAOP_THREADS = 4


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
    torch.set_num_threads(CPU_INTRAOP_THREADS)

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
            num_workers=min(MAX_IMAGE_EMBEDDING_WORKERS, os.cpu_count() or 1),
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

            # FiftyOne's UMAP defaults to init="spectral", which segfaults via ARPACK at this scale.
            label_ids = list(patch_embeddings.keys())
            embeddings = np.stack(list(patch_embeddings.values()))
            points = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                metric="euclidean",
                min_dist=0.1,
                random_state=0,
                init="pca",
                low_memory=True,
                verbose=True,
            ).fit_transform(embeddings)

            fob.compute_visualization(
                dataset,
                patches_field=patches_field,
                points=dict(zip(label_ids, points)),
                method="manual",
                brain_key=PATCH_EMBEDDINGS_KEY,
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
        Dict mapping patch_id -> embedding vector. Patches with invalid geometry are
        absent entirely (no embedding is computed for them).
    """
    patch_embeddings: dict[str, np.ndarray] = {}
    crop_buffer: list[Image.Image] = []
    patch_id_buffer: list[str] = []

    def _embed_buffer() -> None:
        batch_embeds = model.embed_all(crop_buffer)

        # Convert to numpy array if needed
        if hasattr(batch_embeds, "cpu"):
            batch_embeds = batch_embeds.cpu().numpy()
        elif not isinstance(batch_embeds, np.ndarray):
            batch_embeds = np.array(batch_embeds)

        for patch_id, embedding in zip(patch_id_buffer, batch_embeds):
            patch_embeddings[patch_id] = embedding

    crop_stream = iter_patch_crops(
        dataset=dataset,
        patches_field=patches_field,
        dataset_task=dataset_task,
        background_color=(114, 114, 114),
        mask_background=mask_background,
    )

    for _, crops in tqdm(crop_stream, desc="Computing embeddings"):
        crop_buffer.extend(Image.fromarray(crop) for _, crop in crops)
        patch_id_buffer.extend(patch_id for patch_id, _ in crops)

        while len(crop_buffer) >= batch_size:
            crop_buffer, remainder = crop_buffer[:batch_size], crop_buffer[batch_size:]
            patch_id_buffer, patch_id_remainder = patch_id_buffer[:batch_size], patch_id_buffer[batch_size:]
            _embed_buffer()
            crop_buffer, patch_id_buffer = remainder, patch_id_remainder

    if crop_buffer:
        _embed_buffer()

    if not patch_embeddings:
        logger.warning("No crops extracted from dataset")
        return {}

    logger.info(f"Successfully computed embeddings for {len(patch_embeddings)} patches")

    return patch_embeddings
