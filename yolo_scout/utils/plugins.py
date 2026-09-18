"""Plugin management for YOLO Dataset Quality Analysis Tool."""

import shutil
from pathlib import Path

import fiftyone as fo
import yaml

from yolo_scout.utils.logger import logger

_PLUGINS_DIR = Path(__file__).parents[1] / "plugins"


def ensure_plugins() -> None:
    """Copy every local plugin into FiftyOne's plugin directory."""
    if not _PLUGINS_DIR.is_dir():
        # Distributions that don't bundle non-package data (e.g. a stripped wheel) won't
        # have this directory - skip installation instead of crashing the pipeline.
        logger.debug(f"No plugins directory found at {_PLUGINS_DIR}, skipping plugin installation")
        return

    for src in sorted(_PLUGINS_DIR.iterdir()):
        if (src / "fiftyone.yml").exists():
            _ensure_plugin(src.resolve())


def _ensure_plugin(src: Path) -> None:
    """Copy a single plugin into FiftyOne's plugin directory if not already up to date."""
    dst = Path(fo.config.plugins_dir).expanduser().resolve() / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)

    plugin_yml = yaml.safe_load((src / "fiftyone.yml").read_text())
    plugin_name = plugin_yml.get("name", src.name)
    src_version = plugin_yml.get("version")

    dst_yml = dst / "fiftyone.yml"
    if dst.is_symlink():
        dst.unlink()
    elif dst_yml.exists():
        dst_version = yaml.safe_load(dst_yml.read_text()).get("version")
        if dst_version == src_version:
            logger.debug(f"🔌 {plugin_name} already installed — skipping")
            return
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    logger.debug(f"🔌 {plugin_name} v{src_version} installed")
