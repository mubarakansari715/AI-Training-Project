"""Tests for video/audio parsing.

Media files are generated on the fly (a silent WAV via the stdlib, a tiny
MP4 via PyAV) so metadata reading is real; the Whisper model is replaced
with a fake so tests never download a model.
"""

import io
import wave
from types import SimpleNamespace

import av
import numpy as np
import pytest

from app.rag import media_transcriber
from app.rag.document_loader import DocumentParsingError, parse_document


def _wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


def _mp4_bytes(tmp_path, seconds: int = 1, fps: int = 10, with_audio: bool = True) -> bytes:
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=fps)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = container.add_stream("aac", rate=16000) if with_audio else None
        for _ in range(seconds * fps):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio is not None:
            samples = np.zeros((1, 1024), dtype=np.float32)
            for i in range(seconds * 16000 // 1024):
                frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
                frame.sample_rate, frame.pts = 16000, i * 1024
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    return path.read_bytes()


def _segment(start: float, text: str) -> dict:
    return {"start": start, "text": text}


@pytest.fixture
def fake_whisper(monkeypatch):
    """Replace the Whisper worker process; set `.segments` to control its output."""
    fake = SimpleNamespace(segments=[], calls=0)

    def transcribe(media_path, filename):
        fake.calls += 1
        return fake.segments, "en"

    monkeypatch.setattr(media_transcriber, "_transcribe", transcribe)
    return fake


def test_video_becomes_info_page_plus_timestamped_transcript(tmp_path, fake_whisper):
    fake_whisper.segments = [_segment(0.4, " Hello and welcome."), _segment(65.2, " Next topic.")]

    pages = parse_document("demo.mp4", _mp4_bytes(tmp_path))

    info, transcript = pages
    assert all(p.source == "demo.mp4" and p.page is None for p in pages)
    assert "Video file information for 'demo.mp4'" in info.text
    assert "Resolution: 64x48" in info.text
    assert "Has audio: yes" in info.text
    assert "Spoken language: en" in info.text
    assert transcript.text == "[00:00] Hello and welcome.\n[01:05] Next topic."


def test_audio_file_is_labeled_audio_with_duration(fake_whisper):
    fake_whisper.segments = [_segment(0.0, "Just audio.")]

    info, transcript = parse_document("note.wav", _wav_bytes(seconds=2.0))

    assert info.text.startswith("Audio file information for 'note.wav'")
    assert "Duration: 00:02" in info.text
    assert "Resolution" not in info.text
    assert transcript.text == "[00:00] Just audio."


def test_long_transcripts_are_split_into_multiple_pages(fake_whisper):
    fake_whisper.segments = [_segment(t, f"Line at {t}.") for t in range(0, 300, 30)]

    pages = parse_document("talk.wav", _wav_bytes())

    transcript_pages = pages[1:]
    assert len(transcript_pages) == 3  # 0-119s, 120-239s, 240s+
    assert transcript_pages[1].text.startswith("[02:00] Line at 120.")


def test_media_without_speech_still_returns_info_page(fake_whisper):
    fake_whisper.segments = [_segment(1.0, "   ")]

    pages = parse_document("silent.wav", _wav_bytes())

    assert len(pages) == 1
    assert "no speech was detected" in pages[0].text


def test_video_without_audio_track_skips_transcription(tmp_path, fake_whisper):
    pages = parse_document("mute.mp4", _mp4_bytes(tmp_path, with_audio=False))

    assert fake_whisper.calls == 0
    assert "Has audio: no" in pages[0].text


def test_corrupt_media_raises_friendly_error(fake_whisper):
    with pytest.raises(DocumentParsingError, match="Could not read 'broken.mp4'"):
        parse_document("broken.mp4", b"definitely not a video")


def test_format_time_handles_hours():
    assert media_transcriber._format_time(3725.9) == "1:02:05"


def test_worker_failure_raises_friendly_error(monkeypatch):
    import subprocess

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="boom")

    monkeypatch.setattr(media_transcriber.subprocess, "run", fail)

    with pytest.raises(DocumentParsingError, match="Could not transcribe 'note.wav'"):
        parse_document("note.wav", _wav_bytes())


def test_transcribe_speech_joins_spoken_segments(fake_whisper):
    fake_whisper.segments = [_segment(0.0, " What is my"), _segment(1.5, "vitamin D level? "), _segment(3.0, "  ")]

    assert media_transcriber.transcribe_speech(_wav_bytes()) == "What is my vitamin D level?"


def test_transcribe_speech_returns_empty_string_for_silence(fake_whisper):
    assert media_transcriber.transcribe_speech(_wav_bytes()) == ""
