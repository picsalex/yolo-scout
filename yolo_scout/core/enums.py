from enum import Enum


class DatasetTask(Enum):
    """Enumeration of different annotation types."""

    CLASSIFICATION = "classify"
    DETECTION = "detect"
    SEGMENTATION = "segment"
    POSE = "pose"
    OBB = "obb"

    @classmethod
    def is_valid_value(cls, value: str) -> bool:
        """Check if value is a valid enum value."""
        try:
            cls(value.lower())
            return True
        except ValueError:
            return False


class EmbeddingsModel(Enum):
    """Enumeration of different embeddings models."""

    # CLIP
    OPENAI_CLIP = "openai_clip"

    # METACLIP
    METACLIP_400M = "metaclip_400m"
    METACLIP_FULL = "metaclip_fullcc"

    # SIGLIP
    SIGLIP_BASE_224 = "siglip_base_224"

    def get_model_kwargs(self) -> dict | None:
        """
        Get model_kwargs for OpenCLIP models.
        Returns None for hosted models.
        """
        openclip_configs = {
            EmbeddingsModel.OPENAI_CLIP: {
                "clip_model": "ViT-B-32",
                "pretrained": "openai",
            },
            EmbeddingsModel.METACLIP_400M: {
                "clip_model": "ViT-B-32-quickgelu",
                "pretrained": "metaclip_400m",
            },
            EmbeddingsModel.METACLIP_FULL: {
                "clip_model": "ViT-B-32-quickgelu",
                "pretrained": "metaclip_fullcc",
            },
            EmbeddingsModel.SIGLIP_BASE_224: {
                "clip_model": "hf-hub:timm/ViT-B-16-SigLIP",
                "pretrained": "",
            },
        }
        return openclip_configs.get(self)

    @classmethod
    def is_valid_value(cls, value: str) -> bool:
        """Check if value is a valid enum value."""
        try:
            cls(value.lower())
            return True
        except ValueError:
            return False
