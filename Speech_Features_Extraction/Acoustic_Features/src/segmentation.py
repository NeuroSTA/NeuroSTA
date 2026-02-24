import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from pydub import AudioSegment

from .io import ensure_dir


@dataclass
class Segment:
    speaker: str  # "interviewer" or "participant"
    start: Optional[str]
    end: Optional[str]


TS_RE = re.compile(r"(?P<start>\d{2}:\d{2}:\d{2}-\d+)?\s*(?:-)?\s*(?P<end>\d{2}:\d{2}:\d{2}-\d+)?")

def _parse_timestamp_pair(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts start/end timestamps like 00:01:23-5 from a line.
    Supports missing start or end (None).
    """
    m = TS_RE.search(text)
    if not m:
        return None, None
    return m.group("start"), m.group("end")


def timestamp_to_ms(ts: str, fraction_scale: float = 0.1) -> int:
    """
    Convert 'HH:MM:SS-f' to milliseconds.
    fraction_scale=0.1 means '-5' => 0.5 seconds.
    """
    main, frac = ts.split("-")
    hh, mm, ss = main.split(":")
    base_ms = (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000
    frac_ms = int(float(frac) * fraction_scale * 1000)
    return base_ms + frac_ms


def parse_transcript(transcript_path: str, tag_I: str, tag_P: str) -> List[Segment]:
    """
    Parses a transcript file containing lines starting with I: or P: and timestamps.
    """
    segments: List[Segment] = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for ln in lines:
        if ln.startswith(tag_I):
            start, end = _parse_timestamp_pair(ln)
            segments.append(Segment("interviewer", start, end))
        elif ln.startswith(tag_P):
            start, end = _parse_timestamp_pair(ln)
            segments.append(Segment("participant", start, end))

    return segments


def _fix_missing_timestamps(segments: List[Segment]) -> List[Segment]:
    """
    Minimal repair strategy:
    - If start is None, set it to previous segment's end (if available).
    - If end is None, set it to next segment's start (if available).
    - Drop segments still incomplete after attempts.
    """
    # forward fill start
    prev_end = None
    for s in segments:
        if s.start is None and prev_end is not None:
            s.start = prev_end
        if s.end is not None:
            prev_end = s.end

    # backward fill end from next start
    next_start = None
    for s in reversed(segments):
        if s.end is None and next_start is not None:
            s.end = next_start
        if s.start is not None:
            next_start = s.start

    cleaned = [s for s in segments if (s.start is not None and s.end is not None)]
    return cleaned


def _apply_transition_padding(segments: List[Segment], pad_sec: float, fraction_scale: float) -> List[Segment]:
    """
    If speaker changes from interviewer -> participant, push participant start forward by pad_sec
    (prevents overlap artifacts when timestamps touch).
    """
    if pad_sec <= 0:
        return segments

    pad_ms = int(pad_sec * 1000)
    out: List[Segment] = [segments[0]]

    for prev, cur in zip(segments[:-1], segments[1:]):
        if prev.speaker == "interviewer" and cur.speaker == "participant":
            # shift start forward
            cur_start_ms = timestamp_to_ms(cur.start, fraction_scale=fraction_scale)
            shifted = cur_start_ms + pad_ms
            # convert back? easier: keep ms separately not string. We'll apply at slicing time.
            # We'll store as special string "MS:<int>" to avoid reformatting complexity.
            cur.start = f"MS:{shifted}"
        out.append(cur)

    return out


def _ts_or_ms_to_ms(ts: str, fraction_scale: float) -> int:
    if ts.startswith("MS:"):
        return int(ts.split(":", 1)[1])
    return timestamp_to_ms(ts, fraction_scale=fraction_scale)


def find_session_audio(session_dir: str, exts: List[str]) -> Optional[str]:
    """
    Finds an audio file named like the folder (common pattern) or any audio in folder.
    """
    base = os.path.basename(session_dir.rstrip("/\\"))
    for ext in exts:
        cand = os.path.join(session_dir, base + ext)
        if os.path.exists(cand):
            return cand

    # fallback: first audio found
    for root, _, files in os.walk(session_dir):
        for fn in files:
            if any(fn.lower().endswith(e) for e in exts):
                return os.path.join(root, fn)
    return None


def export_merged_speaker_audio(
    audio_path: str,
    segments: List[Segment],
    out_dir: str,
    session_name: str,
    export_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1,
    fraction_scale: float = 0.1,
) -> Dict[str, str]:
    """
    Exports merged interviewer and participant audio into out_dir/{speaker}/.
    Returns dict speaker -> output file path.
    """
    ensure_dir(out_dir)
    audio = AudioSegment.from_file(audio_path)

    # standardize output
    audio = audio.set_channels(channels).set_frame_rate(sample_rate)

    merged = {"interviewer": AudioSegment.empty(), "participant": AudioSegment.empty()}

    for s in segments:
        start_ms = _ts_or_ms_to_ms(s.start, fraction_scale)
        end_ms = _ts_or_ms_to_ms(s.end, fraction_scale)
        if end_ms <= start_ms:
            continue
        merged[s.speaker] += audio[start_ms:end_ms]

    out_paths = {}
    for speaker, seg_audio in merged.items():
        if len(seg_audio) == 0:
            continue
        speaker_dir = os.path.join(out_dir, speaker)
        ensure_dir(speaker_dir)
        out_path = os.path.join(speaker_dir, f"{session_name}_{speaker}.{export_format}")
        seg_audio.export(out_path, format=export_format)
        out_paths[speaker] = out_path

    return out_paths


def segment_all_sessions(paths_cfg: dict, seg_cfg: dict) -> None:
    transcripts_dir = paths_cfg["transcripts_dir"]
    segments_dir = paths_cfg["segments_dir"]
    raw_audio_dir = paths_cfg.get("raw_audio_dir", ".")

    tag_I = seg_cfg["speaker_tags"]["interviewer"]
    tag_P = seg_cfg["speaker_tags"]["participant"]
    fraction_scale = seg_cfg["timestamp"]["fraction_scale"]
    pad_sec = float(seg_cfg.get("transition_padding_sec", 0.0))
    export_format = seg_cfg["export"]["format"]
    sr = int(seg_cfg["export"]["sample_rate"])
    ch = int(seg_cfg["export"]["channels"])
    exts = [e.lower() for e in seg_cfg.get("audio_extensions", [".mp3", ".wav"])]

    sessions = []
    for entry in os.listdir(transcripts_dir):
        session_path = os.path.join(transcripts_dir, entry)
        if os.path.isdir(session_path):
            txts = [f for f in os.listdir(session_path) if f.lower().endswith(".txt")]
            if txts:
                sessions.append((entry, os.path.join(session_path, txts[0])))

    if not sessions:
        raise RuntimeError(f"No transcript sessions found in: {transcripts_dir}")

    for session_name, transcript_path in sessions:
        # locate audio: either in raw_audio_dir/session_name/ or inside transcripts session folder
        candidate_dir = os.path.join(raw_audio_dir, session_name)
        audio_path = find_session_audio(candidate_dir, exts) or find_session_audio(
            os.path.dirname(transcript_path), exts
        )
        if audio_path is None:
            print(f"[segmentation] SKIP {session_name}: no audio found")
            continue

        try:
            segs = parse_transcript(transcript_path, tag_I=tag_I, tag_P=tag_P)
            segs = _fix_missing_timestamps(segs)
            segs = _apply_transition_padding(segs, pad_sec=pad_sec, fraction_scale=fraction_scale)

            out = export_merged_speaker_audio(
                audio_path=audio_path,
                segments=segs,
                out_dir=segments_dir,
                session_name=session_name,
                export_format=export_format,
                sample_rate=sr,
                channels=ch,
                fraction_scale=fraction_scale,
            )
            print(f"[segmentation] OK {session_name}: {list(out.keys())}")
        except Exception as e:
            print(f"[segmentation] ERROR {session_name}: {e}")