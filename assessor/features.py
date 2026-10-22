from __future__ import annotations

import re

import numpy as np

from .types import Segment, fmt

WORD = re.compile(r"[\w']+")
PRAISE = re.compile(
    r"\b(good job|well done|great job|very good|good work|excellent|awesome|amazing|perfect|fantastic|"
    r"nice work|nicely done|that'?s right|exactly|brilliant|wonderful|super|you did it|nice one|great)\b", re.I)
CORRECT = re.compile(r"\b(not quite|almost|try again|think again|let'?s check|have another look|close, but|not exactly)\b", re.I)
FILLER = re.compile(r"\b(um+|uh+|er+|erm|you know|okay okay|so so)\b", re.I)
TECH = re.compile(
    r"\b(can you hear me|can'?t hear|cannot hear|you'?re on mute|unmute|you are on mute|audio|lag|freez\w*|"
    r"frozen|connection|internet|can you see my screen|share my screen|reconnect|you'?re breaking up)\b", re.I)
OPEN_Q = re.compile(
    r"^\s*(what|why|how|which|tell me|explain|describe|what do you think|can you tell me|can you explain|"
    r"how would|why do you think)\b", re.I)
PROMPT = re.compile(r"\b(your turn|can you|do you|what do you think|let'?s try|tell me|show me|who can|anyone)\b", re.I)


def words(t: str) -> int:
    return len(WORD.findall(t))


# ---------------------------------------------------------------- speakers
def assign_speakers(segments: list[Segment], y: np.ndarray, sr: int):
    """Teacher vs student(s) by clustering voice timbre (MFCCs) of each segment.
    The speaker who talks most is the teacher. Falls back to all-teacher when the split is not convincing."""
    info = {"method": "none", "confident": False, "silhouette": None}
    for s in segments:
        s.speaker = "teacher"
    if len(segments) < 8:
        return segments, info
    try:
        import librosa
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        X = []
        for s in segments:
            a = int(s.start * sr)
            b = int(min(s.end, s.start + 8) * sr)
            clip = y[a:b]
            if len(clip) < int(0.4 * sr):
                clip = np.pad(clip, (0, int(0.4 * sr) - len(clip)))
            m = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=20)
            X.append(np.concatenate([m.mean(1), m.std(1)]))
        X = StandardScaler().fit_transform(np.array(X))
        km = KMeans(2, n_init=10, random_state=0).fit(X)
        sil = float(silhouette_score(X, km.labels_))
        dur = [sum(s.end - s.start for s, l in zip(segments, km.labels_) if l == c) for c in (0, 1)]
        counts = [int((km.labels_ == c).sum()) for c in (0, 1)]
        teacher = int(np.argmax(dur))
        info.update(method="mfcc-kmeans", silhouette=round(sil, 3))
        if sil >= 0.08 and min(counts) >= 3:
            for s, l in zip(segments, km.labels_):
                s.speaker = "teacher" if l == teacher else "student"
            info["confident"] = True
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    return segments, info


# ---------------------------------------------------------------- helpers
def _turns(segs: list[Segment], gap: float = 3.0):
    turns = []
    for s in segs:
        if turns and turns[-1]["speaker"] == s.speaker and s.start - turns[-1]["end"] <= gap:
            turns[-1]["end"] = s.end
            turns[-1]["text"] += " " + s.text
        else:
            turns.append({"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text})
    for t in turns:
        t["words"] = words(t["text"])
    return turns


def _audio_features(y: np.ndarray, sr: int, segs: list[Segment]) -> dict:
    import librosa

    f = {}
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    f["speech_to_background_db"] = float(20 * np.log10((np.percentile(rms, 90) + 1e-6) / (np.percentile(rms, 10) + 1e-6)))
    f["clipping_ratio"] = float(np.mean(np.abs(y) > 0.99))
    # dropouts: digital silence >= 1.5s
    dead = rms < 1e-5
    hop_s = 512 / sr
    run, drops = 0, 0
    for d in dead:
        run = run + 1 if d else 0
        if run * hop_s >= 1.5 and run * hop_s - hop_s < 1.5:
            drops += 1
    f["audio_dropouts"] = drops
    # teacher pitch / intonation range (semitones, p90-p10) on a sample of teacher segments
    teacher = [s for s in segs if s.speaker == "teacher" and s.end - s.start > 1.0]
    if teacher:
        idx = np.linspace(0, len(teacher) - 1, min(40, len(teacher))).astype(int)
        semis = []
        for i in idx:
            s = teacher[i]
            clip = y[int(s.start * sr): int(min(s.end, s.start + 6) * sr)]
            if len(clip) < 2048:
                continue
            f0 = librosa.yin(clip, fmin=70, fmax=500, sr=sr, frame_length=1024)
            r = librosa.feature.rms(y=clip, frame_length=1024, hop_length=256)[0][: len(f0)]
            f0 = f0[: len(r)]
            keep = r > 0.4 * np.median(r)
            if keep.sum() > 5:
                semis.append(12 * np.log2(f0[keep] / np.median(f0[keep])))
        if semis:
            allv = np.concatenate(semis)
            f["teacher_pitch_range_st"] = float(np.percentile(allv, 90) - np.percentile(allv, 10))
    return f


def _video_features(frames, step: float, minutes: float) -> dict:
    f = {}
    if len(frames) < 4:
        return f
    diffs = np.array([float(np.mean(np.abs(frames[i][1] - frames[i - 1][1]))) for i in range(1, len(frames))])
    changed = diffs > 8.0
    f["screen_changes_per_min"] = float(changed.sum() / max(minutes, 1e-6))
    f["screen_active_ratio"] = float((diffs > 1.5).mean())
    run, best, best_end = 0, 0, 0
    for i, d in enumerate(diffs):
        run = run + 1 if d < 1.5 else 0
        if run > best:
            best, best_end = run, i
    f["longest_static_screen_s"] = float(best * step)
    f["longest_static_screen_at"] = float(frames[best_end + 1 - best][0]) if best else 0.0
    return f


# ---------------------------------------------------------------- main
def compute_features(segs: list[Segment], y: np.ndarray, sr: int, frames=None, frame_step: float = 10.0,
                     diar_confident: bool = True) -> dict:
    dur = len(y) / sr
    minutes = dur / 60
    f: dict = {"duration_s": dur, "n_segments": len(segs), "diarization_confident": diar_confident}
    T = [s for s in segs if s.speaker == "teacher"]
    S = [s for s in segs if s.speaker == "student"]
    tdur, sdur = sum(s.end - s.start for s in T), sum(s.end - s.start for s in S)
    tw, sw = sum(words(s.text) for s in T), sum(words(s.text) for s in S)
    f.update(teacher_words=tw, student_words=sw, speech_ratio=(tdur + sdur) / dur if dur else 0,
             student_share=sdur / (tdur + sdur) if tdur + sdur else 0,
             teacher_wpm=tw / (tdur / 60) if tdur > 5 else 0.0)

    turns = _turns(segs)
    t_turns = [t for t in turns if t["speaker"] == "teacher"]
    s_turns = [t for t in turns if t["speaker"] == "student"]
    f["student_turns"] = len(s_turns)
    f["student_turns_per_min"] = len(s_turns) / minutes if minutes else 0
    f["avg_student_turn_words"] = float(np.mean([t["words"] for t in s_turns])) if s_turns else 0.0
    f["one_word_answer_share"] = float(np.mean([t["words"] <= 2 for t in s_turns])) if s_turns else 0.0
    ev: dict = {}
    if t_turns:
        lm = max(t_turns, key=lambda t: t["end"] - t["start"])
        f["longest_teacher_monologue_s"] = lm["end"] - lm["start"]
        ev["monologue"] = (lm["start"], lm["end"] - lm["start"], lm["text"][:140])
    f["teacher_turns_over_90s"] = sum(1 for t in t_turns if t["end"] - t["start"] > 90)

    # questions and responses
    q_n, open_n, answered, own_answered, waits, qs = 0, 0, 0, 0, [], []
    for i, s in enumerate(segs):
        if s.speaker != "teacher" or "?" not in s.text:
            continue
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", s.text) if x.strip().endswith("?")]
        q_n += len(sentences) or 1
        open_n += sum(bool(OPEN_Q.match(x)) for x in sentences)
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        outcome = "unanswered"
        wait = None
        if nxt is not None:
            if nxt.speaker == "student" and nxt.start - s.end < 15:
                answered += 1
                wait = max(0.0, nxt.start - s.end)
                waits.append(wait)
                outcome = "answered"
            elif nxt.speaker == "teacher" and nxt.start - s.end < 3:
                own_answered += 1
                outcome = "teacher continued"
        qs.append((s.start, (sentences[-1] if sentences else s.text)[:120], outcome, wait))
    f["questions"] = q_n
    f["questions_per_min"] = q_n / minutes if minutes else 0
    f["open_question_share"] = open_n / q_n if q_n else 0.0
    f["question_response_rate"] = answered / max(len([1 for q in qs]), 1) if qs else 0.0
    f["teacher_answers_own_question_share"] = own_answered / len(qs) if qs else 0.0
    f["median_wait_s"] = float(np.median(waits)) if waits else None
    ev["questions"] = qs[:6]

    # dead air
    gaps = [(segs[i].end, segs[i + 1].start - segs[i].end) for i in range(len(segs) - 1)]
    long = [(a, g) for a, g in gaps if g > 8.0]
    f["dead_air_s"] = float(sum(g for _, g in long))
    f["dead_air_ratio"] = f["dead_air_s"] / dur if dur else 0
    f["dead_air_events"] = len(long)
    if long:
        ev["longest_gap"] = max(long, key=lambda x: x[1])

    # language cues (teacher)
    ttxt = " ".join(s.text for s in T)
    praise = [(s.start, s.text) for s in T if PRAISE.search(s.text)]
    f["praise_per_min"] = len(PRAISE.findall(ttxt)) / minutes if minutes else 0
    f["correction_per_min"] = len(CORRECT.findall(ttxt)) / minutes if minutes else 0
    f["filler_per_100_words"] = 100 * len(FILLER.findall(ttxt)) / max(tw, 1)
    f["engagement_prompts_per_min"] = len(PROMPT.findall(ttxt)) / minutes if minutes else 0
    tech = [(s.start, s.text) for s in segs if TECH.search(s.text)]
    f["tech_issue_mentions"] = len(tech)
    ev["praise"] = praise[:3]
    ev["tech"] = tech[:3]

    # first / last minutes: does the class start and close properly
    f["spoke_in_first_minute"] = bool(any(s.start < 60 for s in segs))

    try:
        f.update(_audio_features(y, sr, segs))
    except Exception as e:  # noqa: BLE001
        f["audio_feature_error"] = str(e)
    if frames:
        f.update(_video_features(frames, frame_step, minutes))
        f["has_video"] = True
    else:
        f["has_video"] = False
    f["_evidence"] = ev
    return f
