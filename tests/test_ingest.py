"""Ingestion tests: probe validation, local paths, and clear failures for bad sources."""
import shutil

import numpy as np
import pytest

from assessor.media import extract_audio, ingest, probe

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not on PATH")


def _write_wav(path, seconds=2.0, sr=16000):
    import soundfile as sf

    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)


@needs_ffmpeg
def test_probe_audio_file(tmp_path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    info = probe(str(wav))
    assert info["has_audio"] is True
    assert info["has_video"] is False
    assert abs(info["duration"] - 2.0) < 0.3


@needs_ffmpeg
def test_ingest_local_path(tmp_path):
    wav = tmp_path / "lesson.wav"
    _write_wav(wav)
    path, info = ingest(str(wav), str(tmp_path / "dl"))
    assert path == str(wav.resolve())
    assert info["has_audio"]


@needs_ffmpeg
def test_ingest_missing_source_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="No such recording"):
        ingest(str(tmp_path / "nope.mp4"), str(tmp_path))


@needs_ffmpeg
def test_ingest_non_media_fails_clearly(tmp_path):
    bogus = tmp_path / "broken.mp4"
    bogus.write_bytes(b"this is not a video " * 100)
    with pytest.raises(RuntimeError, match="not a media file"):
        ingest(str(bogus), str(tmp_path))


@needs_ffmpeg
def test_extract_audio_creates_raw_and_denoised_tracks(tmp_path):
    source = tmp_path / "lesson.wav"
    raw = tmp_path / "raw.wav"
    clean = tmp_path / "clean.wav"
    _write_wav(source)

    extract_audio(str(source), str(raw), denoised_dst=str(clean))

    import soundfile as sf

    raw_audio, raw_sr = sf.read(raw)
    clean_audio, clean_sr = sf.read(clean)
    assert raw_sr == clean_sr == 16000
    assert len(raw_audio) == len(clean_audio)
    assert np.any(clean_audio)


def test_extract_audio_denoises_without_creating_a_second_track(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import assessor.media as media

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        with open(command[-1], "wb") as output_file:
            output_file.write(b"wav")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    output = tmp_path / "clean.wav"
    extract_audio("lesson.mp4", str(output), max_minutes=5, denoise=True)

    assert output.is_file()
    assert calls[0].count("-i") == 1
    assert calls[0][-1] == str(output)
    assert calls[0][calls[0].index("-af") + 1] == "highpass=f=70,afftdn=nf=-25"
