# ruff: noqa: N999 - directory name mirrors the plugin name "@ultralytics/dataset-curation"
"""FiftyOne App operators for curated-subset export and diverse-subset selection."""

import os
import shutil
from datetime import datetime
from pathlib import Path

import fiftyone.operators as foo
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


class SelectDiverseSubset(foo.Operator):
    """Deduplicates near-duplicate clusters while preserving quality variation.

    Groups the dataset into near-duplicate clusters by embedding distance
    (fiftyone.brain's `find_duplicates`), then within each cluster keeps a few
    samples spread evenly across the blurriness range (e.g. sharpest, median,
    blurriest) instead of one arbitrary survivor. This removes redundancy
    without silently discarding the capture-condition variety a model will see
    in production. Samples with no near-duplicates are always kept as-is.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="select_diverse_subset",
            label="Select a diverse subset (dedupe, keep quality variation)",
            dynamic=True,
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.float(
            "threshold",
            default=0.2,
            label="Near-duplicate threshold",
            description="Embedding distance below which two samples are considered near-duplicates",
        )
        inputs.int(
            "per_cluster",
            default=3,
            label="Samples to keep per cluster",
            description="Spread evenly across the blurriness range, e.g. 3 = sharpest, median, blurriest",
        )
        return types.Property(inputs)

    def execute(self, ctx):
        dataset = ctx.dataset
        index = _load_similarity_index(dataset)
        threshold = ctx.params.get("threshold") or 0.2
        per_cluster = max(1, ctx.params.get("per_cluster") or 3)

        index.find_duplicates(thresh=threshold)

        isolated_ids = set(index.unique_ids) - set(index.neighbors_map.keys())
        kept_ids = set(isolated_ids)

        for rep_id, duplicates in index.neighbors_map.items():
            member_ids = [rep_id] + [dup_id for dup_id, _ in duplicates]
            ids, blurs = dataset.select(member_ids).values(["id", "blurriness"])

            order = sorted(range(len(ids)), key=lambda i: (blurs[i] is None, blurs[i]))
            sorted_ids = [ids[i] for i in order]

            n = min(per_cluster, len(sorted_ids))
            positions = {round(i * (len(sorted_ids) - 1) / max(n - 1, 1)) for i in range(n)}
            kept_ids.update(sorted_ids[p] for p in positions)

        view = dataset.select(list(kept_ids))

        # Delegated runs finish long after the App session that queued them may be
        # gone, so there's no live view to push into - save it as a named view
        # instead, and only push it live when we're actually running synchronously.
        view_name = f"diverse_subset_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        dataset.save_view(view_name, view)
        if not ctx.delegated:
            ctx.ops.set_view(view=view)

        return {
            "kept": len(view),
            "total": len(dataset),
            "clusters": len(index.neighbors_map),
            "threshold": threshold,
            "saved_view": view_name,
        }


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
