"""Tests for src.core.enums module."""

import pytest

from yolo_scout.core.enums import DatasetTask, EmbeddingsModel


class TestDatasetTask:
    """Tests for DatasetTask enum."""

    def test_all_task_values_accessible(self):
        """Test that all dataset task values are accessible."""
        assert DatasetTask.CLASSIFICATION.value == "classify"
        assert DatasetTask.DETECTION.value == "detect"
        assert DatasetTask.SEGMENTATION.value == "segment"
        assert DatasetTask.POSE.value == "pose"
        assert DatasetTask.OBB.value == "obb"

    def test_is_valid_value_with_valid_strings(self):
        """Test is_valid_value returns True for valid enum strings."""
        assert DatasetTask.is_valid_value("classify")
        assert DatasetTask.is_valid_value("detect")
        assert DatasetTask.is_valid_value("segment")
        assert DatasetTask.is_valid_value("pose")
        assert DatasetTask.is_valid_value("obb")

        # Test case insensitivity
        assert DatasetTask.is_valid_value("Classify")
        assert DatasetTask.is_valid_value("DETECT")
        assert DatasetTask.is_valid_value("SeGmEnT")

    def test_is_valid_value_with_invalid_strings(self):
        """Test is_valid_value returns False for invalid strings."""
        assert not DatasetTask.is_valid_value("invalid_task")
        assert not DatasetTask.is_valid_value("")
        assert not DatasetTask.is_valid_value("detection")  # wrong spelling

    def test_enum_creation_from_valid_value(self):
        """Test creating enum from valid value string."""
        task = DatasetTask("detect")
        assert task == DatasetTask.DETECTION

    def test_enum_creation_from_invalid_value_raises(self):
        """Test creating enum from invalid value raises ValueError."""
        with pytest.raises(ValueError):
            DatasetTask("invalid")


class TestEmbeddingsModel:
    """Tests for EmbeddingsModel enum."""

    def test_all_models_have_valid_open_clip_configs(self):
        """Test that all embedding models have valid open_clip configurations."""
        try:
            import open_clip
        except ImportError:
            pytest.skip("open_clip not installed")

        # Get all available models from open_clip
        available_models = open_clip.list_pretrained()

        # Test each model in our enum
        for model in EmbeddingsModel:
            kwargs = model.get_model_kwargs()
            assert kwargs is not None, f"{model.name} has no model_kwargs"
            assert "clip_model" in kwargs, f"{model.name} missing 'clip_model' key"
            assert "pretrained" in kwargs, f"{model.name} missing 'pretrained' key"

            clip_model = kwargs["clip_model"]
            pretrained = kwargs["pretrained"]

            # Check if the model/pretrained combination exists in open_clip
            # Note: Some models use hf-hub prefix which won't be in the list
            if not clip_model.startswith("hf-hub:"):
                model_pretrained_pair = (clip_model, pretrained)
                assert model_pretrained_pair in available_models, (
                    f"{model.name}: ({clip_model}, {pretrained}) not found in open_clip available models"
                )

    def test_is_valid_value_with_valid_strings(self):
        """Test is_valid_value returns True for valid model strings."""
        assert EmbeddingsModel.is_valid_value("openai_clip")
        assert EmbeddingsModel.is_valid_value("metaclip_400m")
        assert EmbeddingsModel.is_valid_value("metaclip_fullcc")
        assert EmbeddingsModel.is_valid_value("siglip_base_224")

        # Test case insensitivity
        assert EmbeddingsModel.is_valid_value("OpenAI_CLIP")
        assert EmbeddingsModel.is_valid_value("METACLIP_400M")
        assert EmbeddingsModel.is_valid_value("MeTaClIp_FuLlcc")

    def test_is_valid_value_with_invalid_strings(self):
        """Test is_valid_value returns False for invalid strings."""
        assert not EmbeddingsModel.is_valid_value("invalid_model")
        assert not EmbeddingsModel.is_valid_value("")
        assert not EmbeddingsModel.is_valid_value("clip")

    def test_get_model_kwargs_returns_correct_config(self):
        """Test get_model_kwargs returns correct configuration for each model."""
        # OpenAI CLIP
        kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
        assert kwargs is not None
        assert kwargs["clip_model"] == "ViT-B-32"
        assert kwargs["pretrained"] == "openai"

        # MetaCLIP 400M
        kwargs = EmbeddingsModel.METACLIP_400M.get_model_kwargs()
        assert kwargs is not None
        assert kwargs["clip_model"] == "ViT-B-32-quickgelu"
        assert kwargs["pretrained"] == "metaclip_400m"

        # MetaCLIP Full
        kwargs = EmbeddingsModel.METACLIP_FULL.get_model_kwargs()
        assert kwargs is not None
        assert kwargs["clip_model"] == "ViT-B-32-quickgelu"
        assert kwargs["pretrained"] == "metaclip_fullcc"

        # SigLIP
        kwargs = EmbeddingsModel.SIGLIP_BASE_224.get_model_kwargs()
        assert kwargs is not None
        assert kwargs["clip_model"] == "hf-hub:timm/ViT-B-16-SigLIP"
        assert kwargs["pretrained"] == ""

    def test_model_kwargs_structure(self):
        """Test that model kwargs have expected structure."""
        kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
        assert isinstance(kwargs, dict)
        assert "clip_model" in kwargs
        assert "pretrained" in kwargs
        assert len(kwargs) == 2

    def test_enum_creation_from_valid_value(self):
        """Test creating enum from valid value string."""
        model = EmbeddingsModel("openai_clip")
        assert model == EmbeddingsModel.OPENAI_CLIP

    def test_enum_creation_from_invalid_value_raises(self):
        """Test creating enum from invalid value raises ValueError."""
        with pytest.raises(ValueError):
            EmbeddingsModel("invalid_model")
