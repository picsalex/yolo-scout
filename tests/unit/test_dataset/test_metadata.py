"""Tests for yolo_scout.dataset.metadata module."""

from PIL import Image

from yolo_scout.dataset.metadata import is_image_corrupted


class TestIsImageCorrupted:
    """Tests for is_image_corrupted function."""

    def test_valid_image(self, tmp_path):
        """Test a fully decodable image is not flagged."""
        path = tmp_path / "valid.jpg"
        Image.new("RGB", (64, 64), "red").save(path)

        assert is_image_corrupted(str(path)) is False

    def test_truncated_image(self, tmp_path):
        """Test an image with a readable header but truncated pixel data is flagged."""
        path = tmp_path / "truncated.jpg"
        Image.effect_noise((64, 64), 64).convert("RGB").save(path)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])

        assert is_image_corrupted(str(path)) is True

    def test_missing_file(self, tmp_path):
        """Test a missing file is flagged."""
        assert is_image_corrupted(str(tmp_path / "missing.jpg")) is True
