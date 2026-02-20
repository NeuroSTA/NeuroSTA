from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .pipeline import process_directory


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="speechgraph",
        description="SpeechGraph: graph-based metrics from German transcripts segmented by 'Bild X'.",
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing transcript .txt files.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV outputs (default: outputs/).",
    )
    p.add_argument(
        "--stimuli",
        nargs="+",
        default=["Bild 1", "Bild 2", "Bild 4", "Bild 6"],
        help='Stimulus headings to extract (default: "Bild 1" "Bild 2" "Bild 4" "Bild 6").',
    )
    p.add_argument(
        "--write_stimulus_csvs",
        action="store_true",
        help="Also write separate CSVs per stimulus (e.g., Bild1_data.csv).",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stimuli=args.stimuli,
        write_stimulus_csvs=args.write_stimulus_csvs,
    )

    print(f"Done. Outputs written to: {args.output_dir.resolve()}")
    return 0