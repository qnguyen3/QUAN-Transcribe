"""Rendering of transcription results into OpenAI's response formats."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from .audio import Segment


def _timestamp(seconds: float, *, comma: bool) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def to_srt(segments: Sequence[Segment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    (f"{_timestamp(segment.start, comma=True)} --> "
                    f"{_timestamp(segment.end, comma=True)}"),
                    segment.text,
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(segments: Sequence[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.append(
            f"{_timestamp(segment.start, comma=False)} --> "
            f"{_timestamp(segment.end, comma=False)}"
        )
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines)


def merge_segments_by_sentence(segments: Sequence[Segment]) -> list[Segment]:
    """Join adjacent pieces whose boundary is not a real pause.

    The model splits long audio on energy, not on semantics, so a 90 s file can
    become three pieces with a mid-sentence boundary. When both sides of a
    boundary lack terminal punctuation we report a single segment instead.
    """
    merged: list[Segment] = []
    for segment in segments:
        if merged and _continues(merged[-1], segment):
            previous = merged.pop()
            merged.append(
                Segment(
                    text=f"{previous.text} {segment.text}".strip(),
                    start=previous.start,
                    end=segment.end,
                )
            )
        else:
            merged.append(segment)
    return merged


def _continues(previous: Segment, current: Segment) -> bool:
    if not previous.text or not current.text:
        return False
    if previous.text[-1] in ".?!…:;":
        return False
    # A long silence between pieces is a real boundary even without punctuation.
    return (current.start - previous.end) < 0.5


def transcript_text(segments: Iterable[Segment]) -> str:
    return " ".join(s.text for s in segments if s.text).strip()


def to_verbose_json(
    segments: Sequence[Segment],
    *,
    language: str,
    duration: float,
    model: str,
    task: str = "transcribe",
) -> dict[str, Any]:
    seg_list: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        item: dict[str, Any] = {
            "id": index,
            "seek": 0,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": f" {segment.text}" if segment.text else "",
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 1.0,
            "no_speech_prob": 0.0,
        }
        if getattr(segment, "phonemes", None):
            item["phonemes"] = segment.phonemes
        seg_list.append(item)

    payload: dict[str, Any] = {
        "task": task,
        "language": language,
        "duration": round(duration, 3),
        "text": transcript_text(segments),
        "segments": seg_list,
        "words": [],
        "model": model,
    }
    all_phonemes = " ".join(s.phonemes for s in segments if getattr(s, "phonemes", None)).strip()
    if all_phonemes:
        payload["phonemes"] = all_phonemes
    return payload


def render(
    response_format: str,
    segments: Sequence[Segment],
    *,
    language: str,
    duration: float,
    model: str,
    task: str = "transcribe",
) -> str:
    """Render ``segments`` in the requested OpenAI ``response_format``."""
    fmt = (response_format or "json").lower()
    if fmt == "text":
        return transcript_text(segments)
    if fmt == "srt":
        return to_srt(segments)
    if fmt == "vtt":
        return to_vtt(segments)
    if fmt == "verbose_json":
        return json.dumps(
            to_verbose_json(
                segments,
                language=language,
                duration=duration,
                model=model,
                task=task,
            ),
            ensure_ascii=False,
        )
    if fmt == "json":
        data: dict[str, Any] = {"text": transcript_text(segments)}
        all_phonemes = " ".join(s.phonemes for s in segments if getattr(s, "phonemes", None)).strip()
        if all_phonemes:
            data["phonemes"] = all_phonemes
        return json.dumps(data, ensure_ascii=False)
    raise ValueError(f"unsupported response_format: {response_format!r}")


MEDIA_TYPES = {
    "json": "application/json",
    "verbose_json": "application/json",
    "text": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
}
