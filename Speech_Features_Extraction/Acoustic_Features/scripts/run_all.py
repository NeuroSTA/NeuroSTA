import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_dir", default="configs", help="Path to configs folder")
    ap.add_argument("--skip_segmentation", action="store_true", help="Skip segmentation step")
    args = ap.parse_args()

    if not args.skip_segmentation:
        print("[run_all] Step 1/2: segmentation")
        subprocess.check_call([sys.executable, "scripts/segment_audio.py", "--config_dir", args.config_dir])

    print("[run_all] Step 2/2: feature extraction")
    subprocess.check_call([sys.executable, "scripts/extract_features.py", "--config_dir", args.config_dir])

    print("[run_all] Done.")


if __name__ == "__main__":
    main()