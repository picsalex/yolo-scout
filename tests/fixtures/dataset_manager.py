"""Manage test dataset fixtures by downloading from GitHub releases."""

import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pytest

# Configuration
GITHUB_REPO = "picsalex/yolo-dataset-quality-analysis"
RELEASE_TAG = "v1.0.0"
DATASETS_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/datasets.zip"

# SHA256 checksum for integrity verification
EXPECTED_SHA256 = "a9a770918a03ab05f02b01c35bd80f3f4b650d471c5110a2f1f5215f0d880a96"

FIXTURES_DIR = Path(__file__).parent
CACHE_DIR = FIXTURES_DIR / ".cache"
DATASETS_DIR = FIXTURES_DIR / "datasets"

# Expected dataset structure after extraction
DATASET_PATHS = {
    "detect": "detect_dataset",
    "classify": "classify_dataset",
    "segment": "segment_dataset",
    "pose": "pose_dataset",
    "obb": "obb_dataset",
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    """Verify file integrity using SHA256 checksum."""
    if expected_sha256 is None:
        print("⚠️  Checksum verification disabled (EXPECTED_SHA256 is None)")
        return True

    actual_sha256 = compute_sha256(filepath)
    return actual_sha256 == expected_sha256


def is_datasets_downloaded() -> bool:
    """Check if datasets are already downloaded and extracted."""
    if not DATASETS_DIR.exists():
        return False

    # Verify all expected datasets exist
    for dataset_name in DATASET_PATHS.values():
        dataset_path = DATASETS_DIR / dataset_name
        if not dataset_path.exists():
            return False

    return True


def download_and_extract_datasets(force: bool = False) -> Path:
    """
    Download and extract all test datasets from GitHub releases.

    Args:
        force: Force re-download even if already cached

    Returns:
        Path to datasets directory

    Raises:
        RuntimeError: If download or extraction fails
    """
    # Check if already downloaded
    if is_datasets_downloaded() and not force:
        print(f"✓ Using cached datasets from {DATASETS_DIR}")
        return DATASETS_DIR

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = CACHE_DIR / "datasets.zip"

    # Download if needed
    if not zip_path.exists() or force:
        print(f"📥 Downloading datasets from {DATASETS_URL}...")
        print("   This may take a minute on first run...")

        try:
            urlretrieve(DATASETS_URL, zip_path)
            print(f"✓ Downloaded datasets.zip ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            raise RuntimeError(f"Failed to download datasets.zip: {e}")

    # Verify checksum if configured
    if EXPECTED_SHA256 is not None:
        print("🔍 Verifying checksum...")
        if not verify_checksum(zip_path, EXPECTED_SHA256):
            zip_path.unlink()  # Delete corrupted file
            raise RuntimeError(
                f"Checksum mismatch for datasets.zip. File may be corrupted. "
                f"Expected: {EXPECTED_SHA256}, Got: {compute_sha256(zip_path)}"
            )
        print("✓ Checksum verified")

    # Extract
    print("📦 Extracting datasets...")

    # Clean existing datasets directory
    if DATASETS_DIR.exists():
        shutil.rmtree(DATASETS_DIR)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # List contents for debugging
            file_list = zip_ref.namelist()
            print(f"   Found {len(file_list)} files in archive")

            # Extract all files
            zip_ref.extractall(DATASETS_DIR)

        # Check if extraction created a nested 'datasets' directory
        # (this happens if the ZIP was created with datasets/ as root)
        nested_datasets_dir = DATASETS_DIR / "datasets"
        if nested_datasets_dir.exists() and nested_datasets_dir.is_dir():
            print("   Moving datasets from nested directory...")
            for item in nested_datasets_dir.iterdir():
                dest = DATASETS_DIR / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(DATASETS_DIR))
            # Remove now-empty nested directory
            nested_datasets_dir.rmdir()

        # Verify extraction
        extracted_datasets = []
        for dataset_name in DATASET_PATHS.values():
            dataset_path = DATASETS_DIR / dataset_name
            if dataset_path.exists():
                extracted_datasets.append(dataset_name)
            else:
                print(f"⚠️  Warning: Expected dataset not found: {dataset_name}")

        print(f"✓ Extracted {len(extracted_datasets)} datasets to {DATASETS_DIR}")

        if len(extracted_datasets) != len(DATASET_PATHS):
            raise RuntimeError(f"Expected {len(DATASET_PATHS)} datasets but found {len(extracted_datasets)}")

    except Exception as e:
        if DATASETS_DIR.exists():
            shutil.rmtree(DATASETS_DIR)
        raise RuntimeError(f"Failed to extract datasets.zip: {e}")

    return DATASETS_DIR


def get_dataset_path(dataset_type: str) -> Path:
    """
    Get path to a specific dataset, downloading if necessary.

    Args:
        dataset_type: Type of dataset (detect, classify, segment, pose, obb)

    Returns:
        Path to the dataset directory

    Raises:
        ValueError: If dataset_type is invalid
        RuntimeError: If dataset cannot be found after download
    """
    if dataset_type not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_PATHS.keys())}")

    # Ensure datasets are downloaded
    download_and_extract_datasets()

    dataset_name = DATASET_PATHS[dataset_type]
    dataset_path = DATASETS_DIR / dataset_name

    if not dataset_path.exists():
        raise RuntimeError(f"Dataset {dataset_type} not found at {dataset_path} after download")

    return dataset_path


def clear_cache():
    """Remove all cached datasets and downloads."""
    removed = []

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        removed.append("cache")

    if DATASETS_DIR.exists():
        shutil.rmtree(DATASETS_DIR)
        removed.append("extracted datasets")

    if removed:
        print(f"✓ Cleared: {', '.join(removed)}")
    else:
        print("✓ Nothing to clear")


def print_dataset_info():
    """Print information about downloaded datasets."""
    if not is_datasets_downloaded():
        print("❌ Datasets not downloaded yet")
        print(
            "   Run: python -c 'from tests.fixtures.dataset_manager import download_and_extract_datasets; download_and_extract_datasets()'"
        )
        return

    print(f"📁 Datasets location: {DATASETS_DIR}")
    print("\nAvailable datasets:")

    for dataset_type, dataset_name in DATASET_PATHS.items():
        dataset_path = DATASETS_DIR / dataset_name
        if dataset_path.exists():
            # Count images
            image_dirs = (
                list(dataset_path.glob("**/images/**/*.jpg"))
                + list(dataset_path.glob("**/images/**/*.jpeg"))
                + list(dataset_path.glob("**/images/**/*.png"))
            )

            # For classification, count differently
            if dataset_type == "classify":
                image_dirs = (
                    list(dataset_path.glob("**/*.jpg"))
                    + list(dataset_path.glob("**/*.jpeg"))
                    + list(dataset_path.glob("**/*.png"))
                )

            print(f"  ✓ {dataset_type:10s} ({dataset_name}): {len(image_dirs)} images")
        else:
            print(f"  ✗ {dataset_type:10s} ({dataset_name}): NOT FOUND")


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def datasets_root() -> Path:
    """Provide root directory of all datasets."""
    return download_and_extract_datasets()


@pytest.fixture(scope="session")
def detect_dataset() -> Path:
    """Provide detection dataset fixture."""
    return get_dataset_path("detect")


@pytest.fixture(scope="session")
def classify_dataset() -> Path:
    """Provide classification dataset fixture."""
    return get_dataset_path("classify")


@pytest.fixture(scope="session")
def segment_dataset() -> Path:
    """Provide segmentation dataset fixture."""
    return get_dataset_path("segment")


@pytest.fixture(scope="session")
def pose_dataset() -> Path:
    """Provide pose estimation dataset fixture."""
    return get_dataset_path("pose")


@pytest.fixture(scope="session")
def obb_dataset() -> Path:
    """Provide OBB dataset fixture."""
    return get_dataset_path("obb")


@pytest.fixture(scope="session")
def all_datasets() -> dict[str, Path]:
    """Provide all datasets as a dictionary."""
    download_and_extract_datasets()
    return {dataset_type: get_dataset_path(dataset_type) for dataset_type in DATASET_PATHS}


# ============================================================================
# CLI Utility
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "download":
            download_and_extract_datasets(force="--force" in sys.argv)
        elif command == "clear":
            clear_cache()
        elif command == "info":
            print_dataset_info()
        elif command == "checksum":
            # Compute checksum of downloaded zip
            zip_path = CACHE_DIR / "datasets.zip"
            if zip_path.exists():
                checksum = compute_sha256(zip_path)
                print(f"SHA256: {checksum}")
                print("\nAdd this to dataset_manager.py:")
                print(f'EXPECTED_SHA256 = "{checksum}"')
            else:
                print("❌ datasets.zip not found. Download it first.")
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        print("YOLO Dataset Quality Analysis - Test Dataset Manager")
        print("\nUsage:")
        print("  python dataset_manager.py download [--force]  # Download datasets")
        print("  python dataset_manager.py clear               # Clear cache")
        print("  python dataset_manager.py info                # Show dataset info")
        print("  python dataset_manager.py checksum            # Compute SHA256")
        print("\nOr use as module:")
        print("  from tests.fixtures.dataset_manager import get_dataset_path")
        print("  dataset = get_dataset_path('detect')")
