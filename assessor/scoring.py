from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import fmt

DEFAULT_CONFIG = {
    # label cut-offs on the 1-5 weighted score; tune with calibrate.py on the labelled samples
    "thresholds": {"Poor": 2.6, "Average": 3.4, "Good": 4.1},
    "weights": {
        "Student participation": 0.22,
        "Questioning": 0.14,
        "Responsiveness": 0.10,
        "Delivery & pacing": 0.14,
        "Warmth & feedback": 0.12,
        "Flow & technical quality": 0.10,
        "Screen & visuals": 0.08,
        "Clarity of explanation": 0.10,
        "Rapport with child": 0.10,
        "Content accuracy": 0.10,
    },
}


def load_config(path: str | None = None) -> dict:
    p = Path(path or Path(__file__).resolve().parent.parent / "config.json")
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if p.exists():
        user = json.loads(p.read_text())
        cfg["thresholds"].update(user.get("thresholds", {}))
        cfg["weights"].update(user.get("weights", {}))
    return cfg


def lin(x: float, lo: float, hi: float) -> float:
    """1 at lo, 5 at hi (works when hi < lo too)."""
    if hi == lo:
        return 3.0
    return float(np.clip(1 + 4 * (x - lo) / (hi - lo), 1, 5))


def band(x: float, bad_lo: float, ok_lo: float, ok_hi: float, bad_hi: float) -> float:
    if ok_lo <= x <= ok_hi:
        return 5.0
    return lin(x, bad_lo, ok_lo) if x < ok_lo else lin(x, bad_hi, ok_hi)


def _obs(dim, polarity, text, source="metrics"):
    return {"dimension": dim, "polarity": polarity, "text": text, "source": source}


# ---------------------------------------------------------------- dimensions
def d_participation(f, ev):
    dim = "Student participation"
    if not f["diarization_confident"]:
        return 3.0, [_obs(dim, "-", "Teacher and student voices could not be separated reliably, so participation is "
                                      "scored neutrally and carries less certainty.")]
    obs = []
    share = band(f["student_share"], 0.02, 0.22, 0.60, 0.90)
    turns = lin(f["student_turns_per_min"], 0.2, 2.5)
    depth = lin(f["avg_student_turn_words"], 1.5, 8) * 0.7 + lin(1 - f["one_word_answer_share"], 0.2, 0.7) * 0.3
    score = 0.4 * share + 0.3 * turns + 0.3 * depth
    pct = f["student_share"] * 100
    if f["student_share"] < 0.12:
        obs.append(_obs(dim, "-", f"Students spoke only {pct:.0f}% of the time ({f['student_turns']} turns in "
                                  f"{f['duration_s']/60:.0f} min); this was largely a teacher monologue."))
    elif f["student_share"] <= 0.65:
        obs.append(_obs(dim, "+", f"Students held {pct:.0f}% of the speaking time across {f['student_turns']} turns "
                                  f"(~{f['student_turns_per_min']:.1f} per minute), a healthy balance."))
    else:
        obs.append(_obs(dim, "-", f"Students spoke {pct:.0f}% of the time; the teacher may not be guiding enough."))
    if f["student_turns"] >= 4:
        if f["one_word_answer_share"] > 0.6:
            obs.append(_obs(dim, "-", f"{f['one_word_answer_share']*100:.0f}% of student turns were one or two words; "
                                      "answers were short and not elaborated."))
        elif f["avg_student_turn_words"] >= 6:
            obs.append(_obs(dim, "+", f"Students gave fuller answers (average {f['avg_student_turn_words']:.0f} words per turn)."))
    return score, obs


def d_questioning(f, ev):
    dim = "Questioning"
    obs = []
    qpm = lin(f["questions_per_min"], 0.1, 1.5)
    openq = lin(f["open_question_share"], 0.0, 0.4)
    resp = lin(f["question_response_rate"], 0.2, 0.85)
    own = lin(f["teacher_answers_own_question_share"], 0.5, 0.05)
    score = 0.35 * qpm + 0.2 * openq + 0.25 * resp + 0.2 * own
    if f["questions"] == 0:
        obs.append(_obs(dim, "-", "The teacher asked no questions that were detectable in the transcript."))
        return 1.0, obs
    qs = ev.get("questions", [])
    obs.append(_obs(dim, "+" if f["questions_per_min"] >= 0.6 else "-",
                    f"{f['questions']} teacher questions (~{f['questions_per_min']:.1f}/min); "
                    f"{f['open_question_share']*100:.0f}% were open-ended (what/why/how)."))
    ans = [q for q in qs if q[2] == "answered"]
    una = [q for q in qs if q[2] != "answered"]
    if f["teacher_answers_own_question_share"] > 0.3 and una:
        t, txt, _, _ = una[0]
        obs.append(_obs(dim, "-", f"At {fmt(t)} the teacher asked \"{txt}\" and continued without waiting for a "
                                  f"student reply; this happened for {f['teacher_answers_own_question_share']*100:.0f}% of questions."))
    if ans and f["question_response_rate"] >= 0.5:
        t, txt, _, w = ans[0]
        obs.append(_obs(dim, "+", f"At {fmt(t)} \"{txt}\" drew a student response within {w:.1f}s."))
    return score, obs


def d_responsiveness(f, ev):
    dim = "Responsiveness"
    w = f["median_wait_s"]
    if w is None:
        return 2.0, [_obs(dim, "-", "No question was followed by a student answer, so wait time could not be measured.")]
    score = band(w, 0.0, 1.0, 5.0, 12.0)
    if w < 0.8:
        txt = f"Median pause after questions was {w:.1f}s, which gives children little time to think."
        return score, [_obs(dim, "-", txt)]
    if w <= 5.0:
        return score, [_obs(dim, "+", f"Median wait before a student answered was {w:.1f}s, giving children time to think.")]
    return score, [_obs(dim, "-", f"Median wait for a response was {w:.1f}s; long silences suggest unclear questions or a disengaged student.")]


def d_delivery(f, ev):
    dim = "Delivery & pacing"
    obs = []
    parts = []
    if f["teacher_wpm"] > 0:
        pace = band(f["teacher_wpm"], 60, 105, 165, 230)
        parts.append(pace)
        if f["teacher_wpm"] > 190:
            obs.append(_obs(dim, "-", f"Teacher pace averaged {f['teacher_wpm']:.0f} words per minute, which is fast for children."))
        elif f["teacher_wpm"] < 90:
            obs.append(_obs(dim, "-", f"Teacher pace averaged {f['teacher_wpm']:.0f} wpm; slow, low-energy delivery."))
        else:
            obs.append(_obs(dim, "+", f"Teacher pace of {f['teacher_wpm']:.0f} wpm is comfortable for young learners."))
    if "teacher_pitch_range_st" in f:
        parts.append(lin(f["teacher_pitch_range_st"], 3, 10))
        r = f["teacher_pitch_range_st"]
        obs.append(_obs(dim, "+" if r >= 7 else "-",
                        ("Expressive intonation" if r >= 7 else "Flat, monotone intonation") + f" (pitch range {r:.1f} semitones)."))
    mono = f.get("longest_teacher_monologue_s", 0)
    parts.append(lin(mono, 240, 45))
    m = ev.get("monologue")
    if mono > 90 and m:
        obs.append(_obs(dim, "-", f"Longest uninterrupted teacher talk was {mono:.0f}s starting at {fmt(m[0])} "
                                  f"({f['teacher_turns_over_90s']} stretches over 90s); no student input during that time."))
    if f["filler_per_100_words"] > 3:
        parts.append(2.0)
        obs.append(_obs(dim, "-", f"Frequent filler words ({f['filler_per_100_words']:.1f} per 100 words)."))
    return (float(np.mean(parts)) if parts else 3.0), obs


def d_warmth(f, ev):
    dim = "Warmth & feedback"
    obs = []
    praise = lin(f["praise_per_min"], 0.0, 0.9)
    corr = 5.0 if f["correction_per_min"] > 0.05 else 3.0
    score = 0.7 * praise + 0.3 * corr
    if f["praise_per_min"] >= 0.5:
        ex = ev.get("praise", [])
        extra = f" e.g. at {fmt(ex[0][0])}: \"{ex[0][1][:80]}\"" if ex else ""
        obs.append(_obs(dim, "+", f"Regular encouragement ({f['praise_per_min']:.1f} praise phrases/min){extra}."))
    elif f["praise_per_min"] < 0.15:
        obs.append(_obs(dim, "-", "Very little verbal encouragement or positive reinforcement for the child."))
    if f["correction_per_min"] > 0.05:
        obs.append(_obs(dim, "+", "Teacher gives constructive correction rather than ignoring mistakes."))
    return score, obs


def d_flow(f, ev):
    dim = "Flow & technical quality"
    obs = []
    flow = lin(f["dead_air_ratio"], 0.15, 0.0)
    tech = lin(f["tech_issue_mentions"], 6, 0)
    audio = lin(f.get("speech_to_background_db", 25), 10, 30)
    drop = 1.5 if f.get("audio_dropouts", 0) >= 3 else 5.0
    score = 0.35 * flow + 0.2 * tech + 0.25 * audio + 0.2 * drop
    g = ev.get("longest_gap")
    if f["dead_air_events"] and f["dead_air_ratio"] > 0.04:
        obs.append(_obs(dim, "-", f"{f['dead_air_s']:.0f}s of dead air (silences over 8s, {f['dead_air_events']} times); "
                                  f"longest at {fmt(g[0])} lasting {g[1]:.0f}s."))
    elif f["dead_air_ratio"] < 0.02:
        obs.append(_obs(dim, "+", "The class flowed continuously with almost no dead air."))
    if f["tech_issue_mentions"] >= 2:
        t = ev["tech"][0]
        obs.append(_obs(dim, "-", f"{f['tech_issue_mentions']} mentions of audio/connection trouble (first at {fmt(t[0])}: \"{t[1][:70]}\")."))
    if f.get("audio_dropouts", 0) >= 3:
        obs.append(_obs(dim, "-", f"{f['audio_dropouts']} audio dropouts detected."))
    if f.get("speech_to_background_db", 99) < 14:
        obs.append(_obs(dim, "-", "Persistent background noise makes speech harder to hear."))
    return score, obs


def d_visual(f, ev):
    dim = "Screen & visuals"
    if not f.get("has_video") or "screen_changes_per_min" not in f:
        return None, []
    obs = []
    change = lin(f["screen_changes_per_min"], 0.1, 2.5)
    static = lin(f["longest_static_screen_s"], 600, 90)
    score = 0.5 * change + 0.5 * static
    if f["longest_static_screen_s"] >= 240:
        obs.append(_obs(dim, "-", f"Nothing on screen changed for {f['longest_static_screen_s']/60:.1f} min around "
                                  f"{fmt(f['longest_static_screen_at'])}."))
    if f["screen_changes_per_min"] >= 1.2:
        obs.append(_obs(dim, "+", f"Visuals change about {f['screen_changes_per_min']:.1f} times per minute, keeping attention."))
    elif f["screen_changes_per_min"] < 0.4:
        obs.append(_obs(dim, "-", f"Screen content rarely changed ({f['screen_changes_per_min']:.1f} changes/min)."))
    return score, obs


HEURISTIC_DIMS = [d_participation, d_questioning, d_responsiveness, d_delivery, d_warmth, d_flow, d_visual]


# ---------------------------------------------------------------- assemble
def label_for(score: float, th: dict) -> str:
    if score < th["Poor"]:
        return "Poor"
    if score < th["Average"]:
        return "Average"
    if score < th["Good"]:
        return "Good"
    return "Excellent"


def assess(f: dict, llm: dict | None = None, config: dict | None = None) -> dict:
    cfg = config or load_config()
    ev = f.get("_evidence", {})
    dims, observations = [], []
    if f["n_segments"] == 0 or f["speech_ratio"] < 0.03:
        return {
            "label": "Poor", "score": 1.0, "confidence": "high", "dimensions": [],
            "observations": [_obs("Overall", "-", "Almost no speech was detected in this recording, so it cannot be a functioning class.")],
            "summary": "No meaningful speech detected.", "features": {k: v for k, v in f.items() if not k.startswith("_")},
        }
    for fn in HEURISTIC_DIMS:
        score, obs = fn(f, ev)
        if score is None:
            continue
        name = obs[0]["dimension"] if obs else fn.__name__
        name = {"d_participation": "Student participation", "d_questioning": "Questioning", "d_responsiveness": "Responsiveness",
                "d_delivery": "Delivery & pacing", "d_warmth": "Warmth & feedback", "d_flow": "Flow & technical quality",
                "d_visual": "Screen & visuals"}[fn.__name__]
        dims.append({"name": name, "score": round(score, 2), "weight": cfg["weights"][name], "source": "metrics"})
        observations += obs
    if llm:
        for key, name in [("clarity", "Clarity of explanation"), ("rapport", "Rapport with child"), ("content_accuracy", "Content accuracy")]:
            v = llm.get(key)
            if isinstance(v, (int, float)):
                dims.append({"name": name, "score": float(np.clip(v, 1, 5)), "weight": cfg["weights"][name], "source": "LLM"})
        for pol, k in (("+", "strengths"), ("-", "weaknesses")):
            for item in llm.get(k, [])[:4]:
                ts = item.get("timestamp")
                q = item.get("quote")
                text = (f"At {ts}: " if ts else "") + str(item.get("observation", ""))
                if q:
                    text += f" (\"{str(q)[:110]}\")"
                observations.append(_obs("Teaching content", pol, text, "LLM"))
    wsum = sum(d["weight"] for d in dims)
    score = sum(d["score"] * d["weight"] for d in dims) / wsum
    th = cfg["thresholds"]
    label = label_for(score, th)
    edges = [th["Poor"], th["Average"], th["Good"]]
    margin = min(abs(score - e) for e in edges)
    conf = "high" if margin > 0.3 else "medium" if margin > 0.12 else "low"
    if not f["diarization_confident"] and conf == "high":
        conf = "medium"
    summary = (llm or {}).get("summary") or ""
    observations.sort(key=lambda o: (o["polarity"] != "-", o["dimension"]))
    return {
        "label": label, "score": round(score, 2), "confidence": conf, "dimensions": dims,
        "observations": observations, "summary": summary,
        "features": {k: v for k, v in f.items() if not k.startswith("_")},
    }
