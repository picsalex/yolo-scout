"""Unit tests for src.visualization.quality module."""

import numpy as np
import pytest

from yolo_scout.visualization.quality import _aspect_ratio, _blurriness, _brightness, _entropy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def black_image() -> np.ndarray:
    """Fully black 64x64 grayscale image."""
    return np.zeros((64, 64), dtype=np.uint8)


@pytest.fixture
def white_image() -> np.ndarray:
    """Fully white 64x64 grayscale image."""
    return np.full((64, 64), 255, dtype=np.uint8)


@pytest.fixture
def mid_gray_image() -> np.ndarray:
    """Mid-gray 64x64 grayscale image (value=128)."""
    return np.full((64, 64), 128, dtype=np.uint8)


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """High-contrast alternating black/white 64x64 image."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    return img


@pytest.fixture
def gradient_image() -> np.ndarray:
    """Smooth horizontal gradient 64x64 image — blurry, medium entropy."""
    row = np.linspace(0, 255, 64, dtype=np.uint8)
    return np.tile(row, (64, 1))


# ---------------------------------------------------------------------------
# _brightness
# ---------------------------------------------------------------------------


class TestBrightness:
    def test_black_image_is_zero(self, black_image):
        assert _brightness(black_image) == pytest.approx(0.0, abs=1e-6)

    def test_white_image_is_one(self, white_image):
        assert _brightness(white_image) == pytest.approx(1.0, abs=1e-3)

    def test_mid_gray_is_half(self, mid_gray_image):
        assert _brightness(mid_gray_image) == pytest.approx(128 / 255, abs=1e-3)

    def test_returns_float(self, mid_gray_image):
        assert isinstance(_brightness(mid_gray_image), float)

    def test_range_is_zero_to_one(self, gradient_image):
        val = _brightness(gradient_image)
        assert 0.0 <= val <= 1.0

    def test_brighter_image_has_higher_score(self, black_image, white_image):
        assert _brightness(white_image) > _brightness(black_image)


# ---------------------------------------------------------------------------
# _aspect_ratio
# ---------------------------------------------------------------------------


class TestAspectRatio:
    def test_square_image_is_one(self, black_image):
        """64x64 image → ratio = 1.0."""
        assert _aspect_ratio(black_image) == pytest.approx(1.0, abs=1e-6)

    def test_landscape_is_greater_than_one(self):
        """128x64 image (wider than tall) → ratio > 1."""
        img = np.zeros((64, 128), dtype=np.uint8)
        assert _aspect_ratio(img) == pytest.approx(2.0, abs=1e-6)

    def test_portrait_is_less_than_one(self):
        """64x128 image (taller than wide) → ratio < 1."""
        img = np.zeros((128, 64), dtype=np.uint8)
        assert _aspect_ratio(img) == pytest.approx(0.5, abs=1e-6)

    def test_returns_float(self, black_image):
        assert isinstance(_aspect_ratio(black_image), float)

    def test_is_rounded_to_two_decimals(self):
        """10x3 image → ratio = 3.33."""
        img = np.zeros((3, 10), dtype=np.uint8)
        assert _aspect_ratio(img) == pytest.approx(3.33, abs=0.005)

    def test_content_independent(self, black_image, white_image):
        """Pixel values should not affect the ratio."""
        assert _aspect_ratio(black_image) == _aspect_ratio(white_image)


# ---------------------------------------------------------------------------
# _entropy
# ---------------------------------------------------------------------------


class TestEntropy:
    def test_flat_image_is_zero(self, black_image):
        """Uniform image has a single histogram bin — entropy = 0."""
        assert _entropy(black_image) == pytest.approx(0.0, abs=1e-6)

    def test_flat_white_is_zero(self, white_image):
        assert _entropy(white_image) == pytest.approx(0.0, abs=1e-6)

    def test_checkerboard_has_max_two_bin_entropy(self, checkerboard_image):
        """Two equally likely values → entropy = log2(2) = 1 bit."""
        val = _entropy(checkerboard_image)
        assert val == pytest.approx(1.0, abs=0.05)

    def test_gradient_has_higher_entropy_than_flat(self, black_image, gradient_image):
        assert _entropy(gradient_image) > _entropy(black_image)

    def test_returns_float(self, black_image):
        assert isinstance(_entropy(black_image), float)

    def test_max_entropy_is_bounded(self, gradient_image):
        """Shannon entropy for 256-bin histogram is at most log2(256) = 8 bits."""
        assert _entropy(gradient_image) <= 8.0

    def test_entropy_is_non_negative(self, checkerboard_image):
        assert _entropy(checkerboard_image) >= 0.0


# ---------------------------------------------------------------------------
# _blurriness (regression — ensure existing behaviour unchanged)
# ---------------------------------------------------------------------------


class TestBlurriness:
    def test_flat_image_is_blurry(self, black_image):
        """Flat image has zero Laplacian variance → maximum blurriness (= 1.0)."""
        assert _blurriness(black_image) == pytest.approx(1.0, abs=1e-6)

    def test_checkerboard_is_sharp(self, checkerboard_image):
        """High-frequency image has large Laplacian variance → low blurriness score."""
        assert _blurriness(checkerboard_image) < 0.01

    def test_returns_float(self, black_image):
        assert isinstance(_blurriness(black_image), float)

    def test_sharper_image_has_lower_score(self, black_image, checkerboard_image):
        assert _blurriness(checkerboard_image) < _blurriness(black_image)

    def test_score_is_positive(self, gradient_image):
        assert _blurriness(gradient_image) > 0.0
