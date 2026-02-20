# Acoustic Feature Extraction Pipeline (eGeMAPS / openSMILE)

This repository contains a lightweight, reproducible pipeline for segmenting interview recordings and extracting acoustic/prosodic features using the extended Geneva Minimalistic Acoustic Parameter Set (eGeMAPS) with openSMILE (v3.0). 

## Feature Extraction

### Acoustic & Prosodic Features (n = 88)
Acoustic and prosodic features are computed using openSMILE (version 3.0; Eyben et al., 2010) configured to the extended Geneva Minimalistic Acoustic Parameter Set (eGeMAPS; Eyben et al., 2016). eGeMAPS was designed to provide physiologically interpretable voice parameters relevant for affective and psychiatric research.

The set includes low-level descriptors (LLDs) such as:
- Fundamental frequency (F0) and pitch-related descriptors
- Loudness / energy-related descriptors
- Spectral balance and spectral flux
- Cepstral descriptors (MFCCs 1–4)
- Voice-quality measures (jitter, shimmer, harmonic-to-noise ratio)
- Formant-related measures (F1–F3, bandwidths)

Descriptors are aggregated using statistical functionals (e.g., mean, standard deviation, percentiles, ranges, slopes) to yield one feature vector per recording/participant (depending on configuration).

Preprocessing:
- Signals are amplitude-normalized prior to feature extraction.
- Frame-wise descriptors are aggregated into summary statistics.

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows