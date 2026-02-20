#!/usr/bin/env bash
set -e

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python -m speechgraph \
  --input_dir examples/sample_transcripts \
  --output_dir outputs \
  --stimuli "Bild 1" "Bild 2" "Bild 4" "Bild 6" \
  --write_stimulus_csvs