from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .graphs import NaiveGraph
from .io import process_file
from .metrics import GraphStatistics


@dataclass(frozen=True)
class PipelineOutputs:
    stimulus_level: pd.DataFrame
    participant_level_mean: pd.DataFrame


def _participant_from_filename(filename: str) -> str:
    # Matches your original approach: split at underscore and take first chunk
    return filename.split("_")[0]


def process_directory(
    input_dir: Path,
    output_dir: Path,
    stimuli: Sequence[str],
    write_stimulus_csvs: bool = False,
) -> PipelineOutputs:
    """
    Process all .txt files in input_dir.
    Returns stimulus-level and participant-level averaged DataFrames.
    Also writes CSVs to output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_builder = NaiveGraph()

    rows: List[Dict[str, object]] = []

    for file_path in sorted(input_dir.glob("*.txt")):
        sections = process_file(file_path).sections

        for stim in stimuli:
            text = sections.get(stim, "").strip()

            g = graph_builder.text_to_graph(text)
            stats = GraphStatistics(g).statistics()

            # ID that preserves traceability: <filename>_<BildX>
            # For compatibility with your prior output, also keep Bild without space.
            bild_compact = stim.replace(" ", "")
            record_id = f"{file_path.name}_{bild_compact}"

            row = {"ID": record_id, **stats, "Bild": bild_compact, "Participant": _participant_from_filename(file_path.name)}
            rows.append(row)

    stimulus_df = pd.DataFrame(rows)

    # Write stimulus-level
    stimulus_path = output_dir / "stimulus_level.csv"
    stimulus_df.to_csv(stimulus_path, index=False)

    # Optional separate per-stimulus files
    if write_stimulus_csvs and not stimulus_df.empty:
        for stim in stimuli:
            bild_compact = stim.replace(" ", "")
            sub = stimulus_df[stimulus_df["Bild"] == bild_compact].copy()
            out_path = output_dir / f"{bild_compact}_data.csv"
            sub.to_csv(out_path, index=False)

    # Participant mean across available stimuli
    if stimulus_df.empty:
        participant_df = pd.DataFrame()
    else:
        numeric_cols = [
            "Nodes",
            "Edges",
            "Parallel Edges",
            "Largest Strongly Connected Component (LSC)",
            "Average Total Degree (ATD)",
            "Loops (L1)",
            "Loops (L2)",
            "Loops (L3)",
        ]
        participant_df = (
            stimulus_df.groupby("Participant", as_index=False)[numeric_cols]
            .mean()
        )

    participant_path = output_dir / "participant_level_mean.csv"
    participant_df.to_csv(participant_path, index=False)

    return PipelineOutputs(stimulus_level=stimulus_df, participant_level_mean=participant_df)