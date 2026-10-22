"""Synthetic sanity tests: a well-run interactive class must outscore a monologue with dead air."""
import numpy as np

from calibrate import _monotonicity, fit_dimension_weights, fit_thresholds
from assessor.features import _is_rising_pitch, _yin_pitch, apply_interaction_tags, assign_speakers, compute_features
from assessor.scoring import assess
from assessor.types import Segment

SR = 16000


def audio(seconds):
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(0)
    return (0.1 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.7 * t)) + 0.005 * rng.standard_normal(len(t))).astype(np.float32)


def good_class():
    segs, t = [], 0.0
    for i in range(40):
        segs.append(Segment(t, t + 6, "Great job! What do you think happens next? Can you tell me why?", "teacher")); t += 6.8
        segs.append(Segment(t, t + 5, "I think the plant will grow bigger because it gets water and sun", "student")); t += 5.5
    return segs, t


def poor_class():
    segs, t = [], 0.0
    for i in range(6):
        segs.append(Segment(t, t + 80, "um so this is the lesson and uh you know we read the text " * 6, "teacher")); t += 95
    segs.append(Segment(t, t + 2, "ok", "student"))
    segs.append(Segment(t + 30, t + 33, "Can you hear me? audio is bad", "teacher"))
    return segs, t + 40


def run(segs, dur):
    y = audio(dur)
    f = compute_features(segs, y, SR, None, 10, True)
    return assess(f), f


def test_good_beats_poor():
    g, _ = run(*good_class())
    p, _ = run(*poor_class())
    assert g["score"] > p["score"] + 1.0
    assert p["label"] in ("Poor", "Average")
    assert g["label"] in ("Good", "Excellent")


def test_observations_are_specific():
    p, _ = run(*poor_class())
    text = " ".join(o["text"] for o in p["observations"])
    assert any(ch.isdigit() for ch in text)
    assert "monologue" in text.lower() or "uninterrupted" in text.lower()


def test_silence_is_poor():
    f = compute_features([], audio(60), SR, None, 10, False)
    assert assess(f)["label"] == "Poor"


def test_silence_uses_audio_activity_not_asr_boundaries():
    y = np.zeros(20 * SR, dtype=np.float32)
    t = np.arange(5 * SR) / SR
    voice = (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    y[:5 * SR] = voice
    y[15 * SR:] = voice

    features = compute_features([Segment(0, 20, "Transcript spans the whole file")], y, SR)

    assert features["dead_air_s"] >= 9.0
    assert features["dead_air_ratio"] >= 0.4


def test_yin_pitch_estimates_voiced_tone_without_librosa():
    t = np.arange(SR, dtype=np.float32) / SR
    tone = np.sin(2 * np.pi * 220 * t)

    pitches = _yin_pitch(tone, SR, frame_length=1024)

    assert len(pitches) > 0
    assert abs(float(np.median(pitches)) - 220) < 5


def test_rising_pitch_detection_uses_numpy_estimator():
    t = np.arange(2 * SR, dtype=np.float32) / SR
    rising_tone = np.sin(2 * np.pi * (180 * t + 25 * t * t))

    assert _is_rising_pitch(rising_tone, SR, Segment(0, 2, "statement"))


def test_rising_pitch_detection_ignores_unvoiced_silence():
    assert not _is_rising_pitch(np.zeros(2 * SR, dtype=np.float32), SR, Segment(0, 2, "silence"))


def test_low_confidence_transcript_does_not_drive_lexical_cues():
    features = compute_features(
        [Segment(0, 4, "Great job, excellent work!", confidence=0.1)],
        audio(5),
        SR,
    )

    assert features["praise_per_min"] == 0


def test_threshold_fit_keeps_unobserved_poor_boundary_unchanged():
    current = {"Poor": 2.6, "Average": 3.4, "Good": 3.75}

    fitted = fit_thresholds(
        [3.0, 3.7, 4.2],
        ["Average", "Good", "Excellent"],
        current,
    )

    assert fitted["Poor"] == current["Poor"]
    assert fitted["Poor"] <= fitted["Average"] <= fitted["Good"]


def test_calibration_flags_non_monotone_signals():
    import pandas as pd

    results = _monotonicity(
        pd.DataFrame({"unstable": [1.0, 3.0, 2.0]}),
        pd.Series(["Poor", "Average", "Good"]),
    )

    assert results == [("unstable", "down-weight", 0.5)]


def test_calibration_downweights_non_monotone_dimension():
    rows = [
        {"true": "Poor", "dimensions": {"Questioning": 1.0, "Delivery & pacing": 2.0}},
        {"true": "Average", "dimensions": {"Questioning": 3.0, "Delivery & pacing": 3.0}},
        {"true": "Good", "dimensions": {"Questioning": 2.0, "Delivery & pacing": 4.0}},
    ]

    fitted = fit_dimension_weights(
        rows,
        {"Questioning": 0.2, "Delivery & pacing": 0.1},
    )

    assert fitted == {"Questioning": 0.1, "Delivery & pacing": 0.1}


def test_verbatim_llm_interaction_tags_support_non_english_text():
    features = {
        "duration_s": 60,
        "questions": 0,
        "questions_per_min": 0,
        "praise_per_min": 0,
        "_evidence": {},
    }
    segments = [Segment(0, 5, "നല്ല ജോലി. ഇന്ന് എന്താണ്?", "teacher")]
    tags = {
        "questions": [{"timestamp": "00:03", "quote": "ഇന്ന് എന്താണ്?"}],
        "praise": [{"timestamp": "00:00", "quote": "നല്ല ജോലി"}],
    }

    apply_interaction_tags(features, tags, segments)

    assert features["questions"] == 1
    assert features["questions_per_min"] == 1
    assert features["praise_per_min"] == 1


def test_offline_speaker_embedding_fallback_is_reported(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        "assessor.features._speaker_encoder",
        lambda: (_ for _ in ()).throw(AssertionError("offline mode should not download")),
    )
    segments = [Segment(index, index + 1, "speech") for index in range(8)]

    result, info = assign_speakers(segments, np.zeros(10 * SR, dtype=np.float32), SR)

    assert result == segments
    assert not info["confident"]
    assert "offline mode" in info["error"]
