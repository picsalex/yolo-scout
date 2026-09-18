# ruff: noqa: N999 - directory name mirrors the plugin name "@ultralytics/dataset-curation"
"""FiftyOne App operators for curated-subset export and diverse-subset selection."""

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import fiftyone.operators as foo
import numpy as np
import yaml
from fiftyone.operators import types

from yolo_scout.core.constants import CLASSIFICATION_FIELD, DATASET_SPLITS, SIMILARITY_INDEX_KEY


def _load_similarity_index(dataset):
    if SIMILARITY_INDEX_KEY not in dataset.list_brain_runs():
        raise ValueError(
            f"No similarity index found (brain key '{SIMILARITY_INDEX_KEY}'). "
            "Rerun yolo-scout on this dataset to compute it."
        )
    return dataset.load_brain_results(SIMILARITY_INDEX_KEY)


def _k_center_greedy(embeddings: np.ndarray, quality_scores: np.ndarray, dist_thresh: float) -> list[int]:
    """Greedily selects embeddings that cover the dataset within `dist_thresh`.

    Starts from the highest-quality (lowest quality_scores) sample, then repeatedly
    picks whichever remaining sample is farthest (cosine distance) from everything
    already selected - ties broken by quality - stopping once every remaining sample
    is already within `dist_thresh` of some selected one. Unlike a fixed-radius
    near-duplicate sweep, this can't collapse a long chain of gradually-drifting
    samples (e.g. a panning camera) down to a single survivor: each pick only has to
    clear the growing selected set, not just its immediate neighbors, so the result
    degrades smoothly as `dist_thresh` changes instead of jumping unpredictably.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-12, None)

    seed = int(np.argmin(quality_scores))
    selected = [seed]
    min_dists = 1.0 - normed @ normed[seed]
    min_dists[seed] = -np.inf

    while True:
        farthest = min_dists.max()
        if farthest <= dist_thresh:
            break
        candidates = np.flatnonzero(min_dists == farthest)
        next_idx = int(candidates[np.argmin(quality_scores[candidates])])
        selected.append(next_idx)
        min_dists = np.minimum(min_dists, 1.0 - normed @ normed[next_idx])
        min_dists[next_idx] = -np.inf

    return selected


class SelectDiverseSubset(foo.Operator):
    """Selects a coverage-maximizing subset of the dataset via k-center-greedy.

    See `_k_center_greedy` for the algorithm. The only input is a similarity
    cutoff; how many samples that keeps is a consequence of the data, not a
    number the user has to guess upfront.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="select_diverse_subset",
            label="Select a diverse subset (k-center-greedy)",
            dynamic=True,
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.float(
            "similarity",
            default=0.9,
            required=True,
            label="Similarity threshold",
            description=(
                "0-1, where 1 means identical. Two kept samples are never allowed to be "
                "more similar than this - lower keeps fewer, more diverse samples"
            ),
        )
        return types.Property(inputs)

    def execute(self, ctx):
        dataset = ctx.dataset
        index = _load_similarity_index(dataset)
        similarity = min(max(ctx.params["similarity"], 0.0), 1.0)
        dist_thresh = 1.0 - similarity

        embeddings, sample_ids, _ = index.get_embeddings()
        blurs = dataset.select(list(sample_ids), ordered=True).values("blurriness")
        # Missing scores (e.g. corrupted images skipped during quality metrics) fall
        # back to the worst score, so they never win a farthest-point tie by accident.
        worst = max((b for b in blurs if b is not None), default=0.0)
        quality_scores = np.array([worst if b is None else b for b in blurs])

        selected = _k_center_greedy(embeddings, quality_scores, dist_thresh)
        kept_ids = [sample_ids[i] for i in selected]
        view = dataset.select(kept_ids)

        # Delegated runs finish long after the App session that queued them may be
        # gone, so there's no live view to push into - save it as a named view
        # instead, and only push it live when we're actually running synchronously.
        view_name = f"diverse_subset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        dataset.save_view(view_name, view)
        if not ctx.delegated:
            ctx.ops.set_view(view=view)

        return {"kept": len(view), "total": len(dataset), "similarity": similarity, "saved_view": view_name}


class ExportKeptView(foo.Operator):
    """Copies every sample in the current view to a fresh YOLO dataset on disk.

    Images and label files are copied verbatim from their original source paths -
    no annotation is re-derived, so nothing is lost or altered in the process.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="export_kept_view",
            label="Export current view as a YOLO dataset",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.file(
            "output_dir",
            required=True,
            label="Output directory",
            description="Where to write the curated YOLO dataset",
            view=types.FileExplorerView(choose_dir=True, button_label="Choose a directory..."),
        )
        return types.Property(inputs)

    def execute(self, ctx):
        view = ctx.view if ctx.view is not None else ctx.dataset
        out_dir = Path(ctx.params["output_dir"]["absolute_path"])

        images_copied = 0
        has_labels = False

        for sample in view.iter_samples(progress=False):
            split = next((tag for tag in sample.tags if tag in DATASET_SPLITS), "default")
            image_name = os.path.basename(sample.filepath)

            classification = sample.has_field(CLASSIFICATION_FIELD) and sample[CLASSIFICATION_FIELD] is not None

            if classification:
                class_label = sample[CLASSIFICATION_FIELD].label
                image_dst = out_dir / split / class_label / image_name
                image_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sample.filepath, image_dst)
            else:
                image_dst = out_dir / "images" / split / image_name
                image_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sample.filepath, image_dst)

                if sample.label_path and os.path.exists(sample.label_path):
                    has_labels = True
                    label_dst = out_dir / "labels" / split / os.path.basename(sample.label_path)
                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sample.label_path, label_dst)

            images_copied += 1

        if has_labels:
            class_names = ctx.dataset.info.get("class_names", [])
            (out_dir / "data.yaml").write_text(yaml.safe_dump({"path": ".", "names": dict(enumerate(class_names))}))

        return {"images_copied": images_copied, "output_dir": str(out_dir)}


def register(plugin):
    plugin.register(SelectDiverseSubset)
    plugin.register(ExportKeptView)
