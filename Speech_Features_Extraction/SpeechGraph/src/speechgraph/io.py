from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chardet


_BILD_HEADING_RE = re.compile(r"\n\s*(Bild\s+\d+)\s*\n", re.IGNORECASE)


@dataclass(frozen=True)
class TranscriptSections:
    """Container for extracted stimulus sections."""
    sections: Dict[str, str]  # key: "Bild X" -> value: text


def detect_encoding(file_path: Path) -> str:
    """Detect file encoding via chardet, fallback to utf-8 if unknown."""
    raw = file_path.read_bytes()
    result = chardet.detect(raw)
    enc = result.get("encoding") or "utf-8"
    return enc


def read_text(file_path: Path) -> str:
    """Read text with detected encoding, fallback to latin-1 if needed."""
    enc = detect_encoding(file_path)
    try:
        return file_path.read_text(encoding=enc, errors="strict")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1", errors="replace")


def extract_bild_sections(content: str) -> TranscriptSections:
    """
    Extract sections under headings like "Bild 1", "Bild 2", ...
    Headings must appear on their own line.
    """
    # Split while keeping heading tokens in the result.
    parts = re.split(_BILD_HEADING_RE, content)
    sections: Dict[str, str] = {}

    # parts pattern: [pre, heading1, text1, heading2, text2, ...]
    for i in range(1, len(parts), 2):
        heading = (parts[i] or "").strip()
        text = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        # Preserve original case style: "Bild 1" etc.
        # If duplicates occur, append (safer for interrupted transcripts).
        if heading in sections:
            sections[heading] = (sections[heading] + "\n\n" + text).strip()
        else:
            sections[heading] = text

    return TranscriptSections(sections=sections)


def process_file(file_path: Path) -> TranscriptSections:
    """Read a file and extract 'Bild X' sections."""
    content = read_text(file_path)
    return extract_bild_sections(content)