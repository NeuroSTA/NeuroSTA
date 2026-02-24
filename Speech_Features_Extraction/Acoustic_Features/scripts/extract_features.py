import argparse
import os

from src.io import resolve_config_dir
from src.features import extract_folder_to_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_dir", default="configs", help="Path to configs folder")
    args = ap.parse_args()

    cfg = resolve_config_dir(args.config_dir)
    paths = cfg["paths"]
    feat_cfg = cfg["features"]

    features_dir = paths["features_dir"]
    output_csv = os.path.join(features_dir, feat_cfg.get("output_csv", "egemaps_features.csv"))

    input_source = feat_cfg.get("input_source", "segments")
    if input_source == "segments":
        speaker = feat_cfg.get("segments_speaker", "participant")
        input_dir = os.path.join(paths["segments_dir"], speaker)
    elif input_source == "raw":
        input_dir = paths["raw_audio_dir"]
    else:
        raise ValueError(f"features.yaml: input_source must be 'segments' or 'raw', got {input_source}")

    std = feat_cfg.get("standardize_audio", {})
    extract_folder_to_csv(
        input_dir=input_dir,
        output_csv_path=output_csv,
        feature_set=feat_cfg.get("feature_set", "eGeMAPSv02"),
        feature_level=feat_cfg.get("feature_level", "Functionals"),
        standardize=bool(std.get("enable", True)),
        standardize_sr=int(std.get("sample_rate", 16000)),
        standardize_ch=int(std.get("channels", 1)),
    )


if __name__ == "__main__":
    main()