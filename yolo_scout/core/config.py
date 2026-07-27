"""Configuration management for the application."""

import os
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from yolo_scout.core.enums import DatasetTask, EmbeddingsModel
from yolo_scout.dataset.resolvers import resolve_data_path
from yolo_scout.utils.logger import logger


@dataclass
class Config:
    """Application configuration."""

    # Dataset
    data: str
    name: str
    task: DatasetTask
    reload: bool
    dataset_dir: str

    # Embeddings
    skip_embeddings: bool
    model: EmbeddingsModel
    batch: int
    mask_background: bool

    # Thumbnails
    thumbnail_width: int
    thumbnail_dir: str

    # Quality
    skip_quality: bool

    # App
    port: int
    skip_launch: bool
    verbose: bool

    @classmethod
    def from_cli(cls) -> "Config":
        """Build configuration from CLI args, YAML files, and defaults."""
        args = _parse_arguments()
        config_dict = _build_config_dict(args)
        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, cfg: dict[str, Any]) -> "Config":
        """Create Config instance from flat dictionary."""
        return cls(
            data=cfg["data"],
            name=cfg["name"],
            task=DatasetTask(cfg["task"]),
            reload=_to_bool(cfg["reload"]),
            dataset_dir=cfg["dataset_dir"],
            skip_embeddings=_to_bool(cfg["skip_embeddings"]),
            model=EmbeddingsModel(cfg["model"]),
            batch=int(cfg["batch"]),
            mask_background=_to_bool(cfg["mask_background"]),
            thumbnail_width=int(cfg["thumbnail_width"]),
            thumbnail_dir=cfg["thumbnail_dir"],
            skip_quality=_to_bool(cfg["skip_quality"]),
            port=int(cfg["port"]),
            skip_launch=_to_bool(cfg["skip_launch"]),
            verbose=_to_bool(cfg["verbose"]),
        )


def _load_defaults() -> dict[str, Any]:
    try:
        with files("yolo_scout").joinpath("cfg/default.yaml").open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Default configuration file not found: {e}")
        sys.exit(1)


_DEFAULTS = _load_defaults()
_VALID_ARGS = set(_DEFAULTS.keys()) | {"config"}  # "config" is meta, not in YAML

_HELP_MSG = """yolo-scout commands use the following syntax:

    yolo-scout ARGS

    Where ARGS are any number of 'arg=value' pairs or bare flags:

    Required:
        data=<path>              Path to YOLO dataset directory, data.yaml file,
                                   or ul://username/datasets/slug
        task=<task>              Dataset task: detect | classify | segment | pose | obb

    Optional:
        config=<path>            Path to YAML config file
        name=<str>               FiftyOne dataset name (default: auto from path)
        dataset_dir=<path>        Directory for datasets downloaded via URL (default: yolo_scout/datasets)
        model=<str>              Embeddings model (default: openai_clip)
                                   options: openai_clip | metaclip_400m | metaclip_fullcc | siglip_base_224
        batch=<int>              Batch size for embeddings (default: 16)
        port=<int>               FiftyOne app port (default: 5151)
        thumbnail_dir=<path>     Thumbnail output directory (default: yolo_scout/thumbnails)
        thumbnail_width=<int>    Thumbnail width in pixels (default: 800)
        mask_background=<bool>   Mask background in patch crops (default: true)
        reload                   Force reload dataset even if cached
        skip_embeddings          Skip embedding computation
        skip_quality             Skip quality metrics
        skip_launch              Skip launching FiftyOne app

    Examples:
        yolo-scout data=/path/to/dataset task=detect
        yolo-scout data=/path/to/dataset/data.yaml task=detect
        yolo-scout data=ul://username/datasets/my-dataset task=detect
        yolo-scout data=/path/to/dataset task=detect model=siglip_base_224 batch=32
        yolo-scout config=my_config.yaml batch=8

    Special commands:
        yolo-scout help          Show this message
        yolo-scout version       Print the installed version

    GitHub:   https://github.com/Picsalex/yolo-scout
    PyPI:     https://pypi.org/project/yolo-scout/
    Issues:   https://github.com/Picsalex/yolo-scout/issues
"""


_HELP_FLAGS = {"--help", "-h", "help"}


def handle_special_commands() -> None:
    """Print help or version and exit if a special command is found in sys.argv."""
    if any(t in _HELP_FLAGS for t in sys.argv[1:]):
        print(_HELP_MSG)
        sys.exit(0)

    if any(t == "version" for t in sys.argv[1:]):
        from yolo_scout._version import __version__

        print(f"yolo-scout {__version__}")
        sys.exit(0)


def _parse_arguments() -> dict[str, Any]:
    """Parse key=value style command-line arguments."""
    args = {}
    for token in sys.argv[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            args[k.replace("-", "_")] = v
        else:
            args[token.replace("-", "_")] = True

    unknown = set(args) - _VALID_ARGS
    if unknown:
        bad = ", ".join(f"'{k}'" for k in sorted(unknown))
        received = " ".join(sys.argv)
        print(
            f"\n{bad} is not a valid yolo-scout argument.\nArguments received: {received}\n\n{_HELP_MSG}",
            file=sys.stderr,
        )
        sys.exit(1)

    return args


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes")


def _build_config_dict(args: dict[str, Any]) -> dict[str, Any]:
    """Build configuration dictionary from default.yaml, optional config file, and CLI arguments.
    Precedence (highest last): defaults (cfg/default.yaml) → config file (if provided by the user) → CLI args.
    """
    config = dict(_DEFAULTS)

    if "config" in args:
        config_path = args["config"]
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        with open(config_path, encoding="utf-8") as f:
            config.update(yaml.safe_load(f))  # Overwrite defaults with config file values (if any)

    config.update({k: v for k, v in args.items() if k != "config"})  # Overwrite using the CLI args (if any)

    # Validate and resolve dataset path
    if not config.get("data"):
        logger.error("Dataset path is required (use data=<path> or specify in config file)")
        sys.exit(1)

    try:
        config["data"] = resolve_data_path(
            config["data"], config["dataset_dir"], force=_to_bool(config.get("reload", False))
        )
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
    if not os.path.exists(config["data"]):
        logger.error(f"Dataset path does not exist: {config['data']}")
        sys.exit(1)

    # Validate task
    if not config.get("task"):
        logger.error("Dataset task is required (use task=<task> or specify in config file)")
        sys.exit(1)
    if not DatasetTask.is_valid_value(value=config["task"]):
        logger.error(
            f"Dataset task '{config['task']}' not supported, possible values: {[t.value for t in DatasetTask]}"
        )
        sys.exit(1)

    # Auto-generate dataset name if still set to "default"
    if not config["name"] or config["name"].strip() == "default":
        p = Path(config["data"])
        config["name"] = p.parent.name if not p.is_dir() else p.name
    config["name"] = config["name"].replace(" ", "_")

    # Validate embeddings model
    if not EmbeddingsModel.is_valid_value(value=config.get("model", "")):
        logger.warning(
            f"Embeddings model '{config['model']}' not supported, possible values: {[e.value for e in EmbeddingsModel]}. Defaulting to 'openai_clip'."
        )
        config["model"] = EmbeddingsModel.OPENAI_CLIP.value

    return config
