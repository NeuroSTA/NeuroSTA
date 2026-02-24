import argparse
from src.io import resolve_config_dir
from src.segmentation import segment_all_sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_dir", default="configs", help="Path to configs folder")
    args = ap.parse_args()

    cfg = resolve_config_dir(args.config_dir)
    segment_all_sessions(cfg["paths"], cfg["segmentation"])


if __name__ == "__main__":
    main()