---
tags:
  - Reference
  - Embeddings
---

# Embeddings

YoloScout computes CLIP image embeddings and per-patch embeddings for dataset
visualization and similarity analysis. Embeddings are projected to 2D using UMAP
for interactive exploration in the FiftyOne app.

## Supported models

| Model               | CLI value         | Architecture | Training dataset                                         | Description                                                                                    |
| ------------------- | ----------------- | ------------ | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **OpenAI CLIP**     | `openai_clip`     | ViT-B/32     | [OpenAI CLIP](https://github.com/openai/CLIP)            | Original CLIP model. Hosted on GitHub releases for offline use. Default model.                 |
| **MetaCLIP 400M**   | `metaclip_400m`   | ViT-B/32     | [MetaCLIP](https://github.com/facebookresearch/MetaCLIP) | Trained on curated 400M image-text pairs. Better data quality and embeddings than OpenAI CLIP. |
| **MetaCLIP FullCC** | `metaclip_fullcc` | ViT-B/32     | [MetaCLIP](https://github.com/facebookresearch/MetaCLIP) | Trained on the full CommonCrawl dataset. Highest quality MetaCLIP variant.                     |
| **SigLIP Base**     | `siglip_base_224` | ViT-B/16     | [SigLIP](https://github.com/google-research/big_vision)  | Uses improved sigmoid loss for better performance with smaller batch sizes.                    |

All models use **224x224 input resolution** and produce **512-dimensional
embeddings**. This is a constraint imposed by FiftyOne's OpenCLIP integration —
higher resolution variants (384, 512) cause preprocessing errors. The 224x224
resolution provides excellent quality for most computer vision tasks.

## Model selection guide

- **Use `openai_clip`** if you want the most common, well-known embeddings model
- **Use `metaclip_400m`** for better quality embeddings (recommended default)
- **Use `metaclip_fullcc`** when you need the highest quality embeddings with the
  most diverse training data
- **Use `siglip_base_224`** as an alternative to CLIP-based models with a
  different training objective

All models have similar inference speed and produce embeddings with full support
for FiftyOne visualization and analysis features.

!!! example "Selecting a model"

    ```bash
    # Default: OpenAI CLIP
    yolo-scout data=/path task=detect

    # MetaCLIP for better visual features
    yolo-scout data=/path task=detect model=metaclip_400m

    # Highest quality MetaCLIP variant
    yolo-scout data=/path task=detect model=metaclip_fullcc

    # SigLIP alternative
    yolo-scout data=/path task=detect model=siglip_base_224
    ```

## How it works

### Image embeddings

1. Each image in the dataset is resized and normalized for the selected model
2. The CLIP vision encoder produces a 512-dimensional feature vector
3. UMAP reduces the embeddings to 2D for scatter plot visualization

### Patch embeddings

For tasks with spatial annotations (detect, segment, pose, obb):

1. Each annotation is cropped from its source image
2. Optionally, the background outside the annotation is masked (for segment/OBB)
3. The patch is embedded using the same CLIP model
4. UMAP projects patch embeddings to 2D separately from image embeddings

!!! tip "Background masking"

    For segment and OBB tasks, `mask_background=true` (default) fills the area
    outside the annotation polygon with gray `(114, 114, 114)`. This helps the
    model focus on the object rather than the surrounding context.

    ```bash
    # Disable background masking
    yolo-scout data=/path task=segment mask_background=false
    ```

!!! note

    Patch-level embeddings are not computed for `classify` tasks since there are
    no spatial annotations to crop.

## Batch size

The `batch` parameter controls how many images are processed at once during
embedding computation. Increase it for faster processing on GPUs with more
memory:

```bash
# Default batch size (16)
yolo-scout data=/path task=detect

# Larger batch for GPU with more VRAM
yolo-scout data=/path task=detect batch=64
```

!!! warning

    If you encounter out-of-memory errors, reduce the batch size.

## Skipping embeddings

For a quick dataset overview without embedding computation:

```bash
yolo-scout data=/path task=detect skip_embeddings=true
```

This skips both image and patch embedding computation, making the analysis
significantly faster.

!!! note "First run"

    The first time you use a CLIP model, it will be downloaded automatically.
    Subsequent runs use the cached model.
