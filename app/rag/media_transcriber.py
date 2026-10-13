"""Video/audio parsing via local speech-to-text.

Turns a video or audio file into DocumentPage objects so it can be
indexed and questioned like any other document: one "media info" page
(duration, resolution, frame rate, audio, language) plus the spoken
transcript, one line per segment prefixed with an [mm:ss] timestamp so
answers can say *when* something was said.

Transcription runs locally with faster-whisper on the CPU, in a separate
process (see app.rag.transcribe_worker for why); the model is downloaded
once on first use and then cached. Only speech is captured — what is
shown on screen is not analysed.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config.settings import settings
from app.rag.document_loader import DocumentPage, DocumentParsingError

logger = logging.getLogger(__name__)

# Transcript lines are grouped into pages of this many seconds, so a long
# recording becomes several retrievable units instead of one huge text.
_PAGE_SECONDS = 120
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Generous upper bound so a stuck transcription can't hang the app forever.
_TRANSCRIBE_TIMEOUT_SECONDS = 60 * 60


def _transcribe(media_path: str, filename: str) -> tuple[list[dict], str | None]:
    """Run the Whisper worker process; return (segments, language)."""
    with tempfile.NamedTemporaryFile(suffix=".json") as out:
        try:
            subprocess.run(
                [sys.executable, "-m", "app.rag.transcribe_worker",
                 media_path, settings.whisper_model, out.name],
                cwd=_PROJECT_ROOT, check=True, capture_output=True, text=True,
                timeout=_TRANSCRIBE_TIMEOUT_SECONDS,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Transcription of %s failed:\n%s", filename, exc.stderr[-2000:])
            raise DocumentParsingError(f"Could not transcribe '{filename}'.") from exc
        except subprocess.TimeoutExpired as exc:
            raise DocumentParsingError(f"Transcribing '{filename}' took too long.") from exc
        result = json.loads(Path(out.name).read_text(encoding="utf-8"))
    return result["segments"], result["language"]


def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _read_metadata(path: str) -> dict:
    import av

    with av.open(path) as container:
        meta = {
            "duration": container.duration / 1_000_000 if container.duration else None,
            "has_audio": bool(container.streams.audio),
            "video": None,
        }
        if container.streams.video:
            stream = container.streams.video[0]
            rate = stream.average_rate
            meta["video"] = {
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "fps": round(float(rate), 2) if rate else None,
            }
    return meta


def _info_page(filename: str, meta: dict, language: str | None, segment_count: int) -> DocumentPage:
    kind = "Video" if meta["video"] else "Audio"
    lines = [f"{kind} file information for '{filename}':"]
    if meta["duration"] is not None:
        lines.append(f"- Duration: {_format_time(meta['duration'])} ({meta['duration']:.1f} seconds)")
    if meta["video"]:
        video = meta["video"]
        lines.append(f"- Resolution: {video['width']}x{video['height']}")
        if video["fps"]:
            lines.append(f"- Frame rate: {video['fps']} fps")
    lines.append(f"- Has audio: {'yes' if meta['has_audio'] else 'no'}")
    if language:
        lines.append(f"- Spoken language: {language}")
    if segment_count:
        lines.append(f"- Transcript: {segment_count} spoken segments, timestamps as [mm:ss]")
    else:
        lines.append("- Transcript: no speech was detected")
    lines.append("Only the spoken audio is transcribed; on-screen visuals are not described.")
    return DocumentPage(text="\n".join(lines), source=filename, page=None)


def transcribe_media(data: bytes, filename: str) -> list[DocumentPage]:
    """Return an info page plus timestamped transcript pages for a media file."""
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()

        try:
            meta = _read_metadata(tmp.name)
        except Exception as exc:
            raise DocumentParsingError(
                f"Could not read '{filename}'. The file may be corrupted or use an unsupported codec."
            ) from exc

        segments, language = [], None
        if meta["has_audio"]:
            segments, language = _transcribe(tmp.name, filename)

    spoken = [s for s in segments if s["text"].strip()]
    pages = [_info_page(filename, meta, language, len(spoken))]

    current: list[str] = []
    page_start = 0.0
    for segment in spoken:
        start, text = segment["start"], segment["text"].strip()
        if current and start - page_start >= _PAGE_SECONDS:
            pages.append(DocumentPage(text="\n".join(current), source=filename, page=None))
            current = []
        if not current:
            page_start = start
        current.append(f"[{_format_time(start)}] {text}")
    if current:
        pages.append(DocumentPage(text="\n".join(current), source=filename, page=None))

    return pages

