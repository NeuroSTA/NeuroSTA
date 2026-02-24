import os
import yaml
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_config_dir(config_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads configs from the repo structure:
      configs/paths.yaml
      configs/segmentation.yaml
      configs/features.yaml

    If paths.yaml doesn't exist, falls back to paths.example.yaml.
    """
    paths_yaml = os.path.join(config_dir, "paths.yaml")
    if not os.path.exists(paths_yaml):
        paths_yaml = os.path.join(config_dir, "paths.example.yaml")

    cfg = {
        "paths": load_yaml(paths_yaml),
        "segmentation": load_yaml(os.path.join(config_dir, "segmentation.yaml")),
        "features": load_yaml(os.path.join(config_dir, "features.yaml")),
    }

    # Create output dirs
    outputs_dir = cfg["paths"].get("outputs_dir", "outputs")
    segments_dir = cfg["paths"].get("segments_dir", os.path.join(outputs_dir, "segments"))
    features_dir = cfg["paths"].get("features_dir", os.path.join(outputs_dir, "features"))

    ensure_dir(outputs_dir)
    ensure_dir(segments_dir)
    ensure_dir(features_dir)

    # Speaker subfolders for segments
    ensure_dir(os.path.join(segments_dir, "interviewer"))
    ensure_dir(os.path.join(segments_dir, "participant"))

    return cfg