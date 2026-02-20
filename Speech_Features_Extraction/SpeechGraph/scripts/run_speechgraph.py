from __future__ import annotations

import argparse
from pathlib import Path

from speechgraph.pipeline import process_directory


def main() -> int:
    p = argparse.ArgumentParser(description="Run SpeechGraph on a directory of transcripts.")
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--stimuli",
        nargs="+",
        default=["Bild 1", "Bild 2", "Bild 4", "Bild 6"],
    )
    p.add_argument("--write_stimulus_csvs", action="store_true")
    args = p.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stimuli=args.stimuli,
        write_stimulus_csvs=args.write_stimulus_csvs,
    )
    print(f"Done. Outputs written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())