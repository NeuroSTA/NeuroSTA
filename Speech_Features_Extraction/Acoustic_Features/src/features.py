import os
from typing import List, Optional

import pandas as pd
import opensmile
from pydub import AudioSegment

from .io import ensure_dir


def _get_smile(feature_set: str, feature_level: str) -> opensmile.Smile:
    fs_map = {
        "eGeMAPSv02": opensmile.FeatureSet.eGeMAPSv02,
        "ComParE_2016": opensmile.FeatureSet.ComParE_2016,
    }
    fl_map = {
        "Functionals": opensmile.FeatureLevel.Functionals,
        "LowLevelDescriptors": opensmile.FeatureLevel.LowLevelDescriptors,
    }
    if feature_set not in fs_map:
        raise ValueError(f"Unknown feature_set: {feature_set}. Use one of: {list(fs_map)}")
    if feature_level not in fl_map:
        raise ValueError(f"Unknown feature_level: {feature_level}. Use one of: {list(fl_map)}")

    return opensmile.Smile(feature_set=fs_map[feature_set], feature_level=fl_map[feature_level])


def list_audio_files(input_dir: str, exts: Optional[List[str]] = None) -> List[str]:
    if exts is None:
        exts = [".wav", ".mp3"]
    out = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if any(fn.lower().endswith(e) for e in exts):
                out.append(os.path.join(root, fn))
    return sorted(out)


def standardize_to_wav_mono_16k(in_path: str, out_path: str, sr: int = 16000, ch: int = 1) -> str:
    """
    Uses pydub (ffmpeg) to decode and write standardized WAV.
    Returns out_path.
    """
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_channels(ch).set_frame_rate(sr)
    ensure_dir(os.path.dirname(out_path))
    audio.export(out_path, format="wav")
    return out_path


def extract_folder_to_csv(
    input_dir: str,
    output_csv_path: str,
    feature_set: str = "eGeMAPSv02",
    feature_level: str = "Functionals",
    standardize: bool = True,
    standardize_sr: int = 16000,
    standardize_ch: int = 1,
    tmp_wav_dir: str = "outputs/_tmp_wav",
) -> str:
    """
    Extracts features for every audio file in input_dir and writes a single CSV.
    """
    ensure_dir(os.path.dirname(output_csv_path))
    smile = _get_smile(feature_set=feature_set, feature_level=feature_level)

    files = list_audio_files(input_dir)
    if not files:
        raise RuntimeError(f"No audio files found in: {input_dir}")

    dfs = []
    ensure_dir(tmp_wav_dir)

    for path in files:
        base = os.path.basename(path)
        print(f"[features] Processing: {base}")

        proc_path = path
        if standardize:
            # Always standardize to wav for consistent feature extraction
            tmp_out = os.path.join(tmp_wav_dir, os.path.splitext(base)[0] + ".wav")
            proc_path = standardize_to_wav_mono_16k(path, tmp_out, sr=standardize_sr, ch=standardize_ch)

        feat = smile.process_file(proc_path)

        # add file name
        feat["file"] = base
        cols = ["file"] + [c for c in feat.columns if c != "file"]
        feat = feat[cols]

        dfs.append(feat)

    all_features = pd.concat(dfs, ignore_index=True)
    all_features.to_csv(output_csv_path, index=False)
    print(f"[features] Saved: {output_csv_path}")
    return output_csv_path