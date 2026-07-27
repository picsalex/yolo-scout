"""Tests for src.core.constants module."""

from yolo_scout.core.constants import (
    CLASSIFICATION_FIELD,
    DETECTION_FIELD,
    KEYPOINTS_FIELD,
    OBB_FIELD,
    SEGMENTATION_FIELD,
    get_color_palette,
    get_field_name,
)
from yolo_scout.core.enums import DatasetTask


class TestGetFieldName:
    """Tests for get_field_name function."""

    def test_get_field_name_for_classification(self):
        """Test get_field_name returns correct field for classification."""
        field = get_field_name(DatasetTask.CLASSIFICATION)
        assert field == CLASSIFICATION_FIELD

    def test_get_field_name_for_detection(self):
        """Test get_field_name returns correct field for detection."""
        field = get_field_name(DatasetTask.DETECTION)
        assert field == DETECTION_FIELD

    def test_get_field_name_for_segmentation(self):
        """Test get_field_name returns correct field for segmentation."""
        field = get_field_name(DatasetTask.SEGMENTATION)
        assert field == SEGMENTATION_FIELD

    def test_get_field_name_for_pose(self):
        """Test get_field_name returns correct field for pose."""
        field = get_field_name(DatasetTask.POSE)
        assert field == KEYPOINTS_FIELD

    def test_get_field_name_for_obb(self):
        """Test get_field_name returns correct field for OBB."""
        field = get_field_name(DatasetTask.OBB)
        assert field == OBB_FIELD


class TestGetColorPalette:
    """Tests for get_color_palette function."""

    def test_color_palette_length_matches_labels(self):
        """Test that color palette length matches number of labels."""
        labels = ["cat", "dog", "person"]
        palette = get_color_palette(labels)
        assert len(palette) == 3

    def test_color_palette_with_single_label(self):
        """Test color palette with a single label."""
        labels = ["cat"]
        palette = get_color_palette(labels)
        assert len(palette) == 1
        assert palette[0]["value"] == "cat"
        assert palette[0]["color"].startswith("#")

    def test_color_palette_structure(self):
        """Test that each palette entry has correct structure."""
        labels = ["cat", "dog"]
        palette = get_color_palette(labels)

        for i, entry in enumerate(palette):
            assert "value" in entry
            assert "color" in entry
            assert entry["value"] == labels[i]
            assert isinstance(entry["color"], str)
            assert entry["color"].startswith("#")
            assert len(entry["color"]) == 7  # #RRGGBB

    def test_color_palette_with_many_labels(self):
        """Test color palette with more labels than predefined colors."""
        # Create more labels than ULTRALYTICS_COLORS (20)
        labels = [f"class_{i}" for i in range(30)]
        palette = get_color_palette(labels)

        assert len(palette) == 30
        # First 20 should use predefined colors
        # Remaining 10 should use colormap
        for entry in palette:
            assert "value" in entry
            assert "color" in entry
            assert entry["color"].startswith("#")

    def test_color_palette_hex_format(self):
        """Test that colors are in valid hex format."""
        labels = ["cat", "dog", "bird"]
        palette = get_color_palette(labels)

        for entry in palette:
            color = entry["color"]
            assert color.startswith("#")
            # Check if rest is valid hex
            hex_part = color[1:]
            assert len(hex_part) == 6
            assert all(c in "0123456789ABCDEFabcdef" for c in hex_part)

    def test_color_palette_empty_labels(self):
        """Test color palette with empty labels list."""
        labels = []
        palette = get_color_palette(labels)
        assert len(palette) == 0
        assert palette == []

    def test_color_palette_preserves_label_order(self):
        """Test that color palette preserves label order."""
        labels = ["zebra", "apple", "cat", "dog"]
        palette = get_color_palette(labels)

        for i, entry in enumerate(palette):
            assert entry["value"] == labels[i]
