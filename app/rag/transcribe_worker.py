"""Standalone speech-to-text process for app.rag.media_transcriber.

Runs in its own process on purpose: faster-whisper (CTranslate2) and
PyTorch (used for embeddings) each bundle Intel's OpenMP runtime
(libiomp5), and loading both into one process aborts it. This module
must therefore never import PyTorch or anything from the app that does.

Usage: python -m app.rag.transcribe_worker <media_path> <model_name> <out_json>
Writes {"language": str, "segments": [{"start": float, "text": str}, ...]}.
"""

from __future__ import annotations

import json
import sys


def main(media_path: str, model_name: str, out_path: str) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(media_path, vad_filter=True)
    result = {
        "language": info.language,
        "segments": [{"start": s.start, "text": s.text} for s in segments if s.text.strip()],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main(*sys.argv[1:4])
