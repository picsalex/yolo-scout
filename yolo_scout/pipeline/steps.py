"""Pipeline steps for the yolo-scout curation pipeline."""

import logging
from pathlib import Path

import fiftyone as fo

from yolo_scout.core.config import Config
from yolo_scout.utils.decorators import step


@step
def validate(verbose: bool = True) -> Config:
    config = Config.from_cli()

    if verbose:
        from yolo_scout.utils.logger import logger

        params_str = ", ".join(f"{k}={v.value if hasattr(v, 'value') else v}" for k, v in vars(config).items())
        logger.info(f"configuration: {params_str}")

    return config


@step(name="preparation")
def prepare_run(config: Config) -> bool:
    """Determine cache state and clean up if needed. Returns recompute=True if pipeline should run."""
    from yolo_scout.utils.logger import logger
    from yolo_scout.visualization.thumbnails import delete_thumbnails

    if config.name not in fo.list_datasets():
        logger.info(f"No cache found for '{config.name}', running full pipeline")
        return True

    if config.reload:
        logger.info(f"Reload requested, clearing cache for '{config.name}'")
        fo.delete_dataset(name=config.name)
        delete_thumbnails(dataset_name=config.name, thumbnail_dir=config.thumbnail_dir)
        return True

    dataset = fo.load_dataset(name=config.name)
    if len(dataset) == 0:
        logger.warning(f"Cached dataset '{config.name}' is empty (interrupted run), reloading")
        fo.delete_dataset(name=config.name)
        delete_thumbnails(dataset_name=config.name, thumbnail_dir=config.thumbnail_dir)
        return True

    logger.info(f"Cache hit for '{config.name}' ({len(dataset)} samples), skipping reload")
    return False


@step(name="fiftyone_plugins_setup", level=logging.DEBUG)
def prepare_plugins() -> None:
    from yolo_scout.utils.plugins import ensure_plugins

    ensure_plugins()


@step(name="dataset")
def run_load_dataset(config: Config) -> fo.Dataset:
    from yolo_scout.dataset.loader import load_yolo_dataset

    return load_yolo_dataset(
        config=config,
    )


@step(name="embeddings")
def run_embeddings(dataset: fo.Dataset, config: Config, recompute: bool) -> None:
    from yolo_scout.utils.logger import logger

    if not recompute:
        logger.info("Skipping: dataset loaded from cache")
        return
    if config.skip_embeddings:
        logger.info("Skipping: user requested")
        return

    from yolo_scout.embeddings.computer import compute_embeddings

    compute_embeddings(
        dataset=dataset,
        dataset_task=config.task,
        model_kwargs=config.model.get_model_kwargs(),
        batch_size=config.batch,
        mask_background=config.mask_background,
    )


@step(name="quality_metrics")
def run_quality_metrics(dataset: fo.Dataset, config: Config, recompute: bool) -> None:
    from yolo_scout.utils.logger import logger

    if not recompute:
        logger.info("Skipping: dataset loaded from cache")
        return
    if config.skip_quality:
        logger.info("Skipping: user requested")
        return

    from yolo_scout.visualization.quality import compute_quality_metrics

    compute_quality_metrics(
        dataset=dataset,
        dataset_task=config.task,
        mask_background=config.mask_background,
    )


@step(name="launch")
def run_launch(dataset: fo.Dataset, config: Config) -> None:
    if not config.skip_launch:
        from yolo_scout.visualization.fiftyone_ops import launch_fiftyone_app

        launch_fiftyone_app(dataset=dataset, dataset_task=config.task, port=config.port)
    else:
        from yolo_scout.utils.logger import logger

        logger.info(f"Skipping launch. To open later: fiftyone app launch {config.name}")


@step(name="thumbnails")
def run_thumbnails(dataset: fo.Dataset, config: Config, recompute: bool) -> None:
    from yolo_scout.utils.logger import logger

    if config.thumbnail_width <= 1:
        logger.info(f"Skipping: thumbnail_width={config.thumbnail_width} must be > 1")
        return

    thumbnail_dir = Path(config.thumbnail_dir) / config.name
    cached_width = dataset.info.get("thumbnail_width")
    if not recompute and cached_width == config.thumbnail_width and thumbnail_dir.exists():
        logger.info(f"Skipping: thumbnails ({config.thumbnail_width}px) already exist")
        return

    from yolo_scout.visualization.thumbnails import generate_thumbnails

    generate_thumbnails(
        dataset=dataset,
        thumbnail_dir_path=str(thumbnail_dir.resolve()),
        thumbnail_width=config.thumbnail_width,
    )
