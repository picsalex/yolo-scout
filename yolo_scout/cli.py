"""
YoloScout — YOLO Dataset Quality Analysis Tool
CLI entry point

"""

import os

# Must be set before numba is imported anywhere (its thread pool can't shrink later).
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

from yolo_scout.core.config import handle_special_commands
from yolo_scout.utils.decorators import pipeline
from yolo_scout.utils.logger import configure_external_loggers, logger


@pipeline
def main():
    handle_special_commands()
    configure_external_loggers()

    from yolo_scout.pipeline.steps import (
        prepare_plugins,
        prepare_run,
        run_embeddings,
        run_launch,
        run_load_dataset,
        run_quality_metrics,
        run_thumbnails,
        validate,
    )

    logger.info("=" * 60)
    logger.info("FIFTYONE YOLO DATASET ANALYSIS")
    logger.info("=" * 60)

    config = validate(verbose=True)
    recompute = prepare_run(config=config)

    prepare_plugins()

    dataset = run_load_dataset(config=config)
    run_embeddings(dataset=dataset, config=config, recompute=recompute)
    run_quality_metrics(dataset=dataset, config=config, recompute=recompute)
    run_thumbnails(dataset=dataset, config=config, recompute=recompute)
    run_launch(dataset=dataset, config=config)


if __name__ == "__main__":
    main()
