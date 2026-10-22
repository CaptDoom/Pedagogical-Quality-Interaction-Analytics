"""Synthetic sanity tests: a well-run interactive class must outscore a monologue with dead air."""
import numpy as np

from assessor.features import compute_features
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
