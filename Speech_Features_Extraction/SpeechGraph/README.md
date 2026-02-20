# SpeechGraph

Graph-based analysis of spontaneous speech transcripts (German) using directed word-transition graphs. SpeechGraph splits transcripts into stimulus sections (e.g., “Bild 1”, “Bild 2”, “Bild 4”, “Bild 6”), constructs directed MultiDiGraphs from token sequences, computes graph metrics per stimulus, and exports CSVs at stimulus level and participant-averaged level.

This repo is designed for research workflows (e.g., psychiatric speech analysis, formal thought disorder). It emphasizes transparent feature extraction and traceability from transcript → stimulus section → participant summary.

## Features

- Batch processing of `.txt` transcripts from an input directory
- Robust “Bild X” section extraction using regex
- Encoding detection (chardet) with safe fallbacks
- Directed word-transition graphs (NetworkX `MultiDiGraph`)
- Metrics per stimulus section:
  - Nodes
  - Edges
  - Parallel Edges (transition pairs occurring >1 times)
  - Largest Strongly Connected Component (LSC)
  - Average Total Degree (ATD)
  - Loops (L1/L2/L3) via adjacency matrix powers
- Exports:
  - `stimulus_level.csv` (one row per file × stimulus)
  - `participant_level_mean.csv` (mean across stimuli per participant)
  - optional per-stimulus CSVs (`Bild1_data.csv`, etc.)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt