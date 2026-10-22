from __future__ import annotations

import os
import re
import shutil
from functools import lru_cache
from typing import Callable

import numpy as np

from .types import Segment

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


def _confidence_weight(segment: Segment) -> float:
    return 1.0 if segment.confidence is None else float(np.clip(segment.confidence, 0.1, 1.0))


def _short_time_rms(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(y) < frame:
        return np.empty(0, dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    rms = np.empty(len(windows), dtype=np.float32)
    for start in range(0, len(windows), 10000):
        batch = windows[start:start + 10000].astype(np.float32, copy=False)
        rms[start:start + len(batch)] = np.sqrt(np.mean(batch * batch, axis=1))
    return rms


def _voice_activity_intervals(y: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Estimate speech regions from short-time audio energy, independent of ASR boundaries."""
    frame = max(1, int(sr * 0.03))
    hop = max(1, int(sr * 0.01))
    if len(y) < frame:
        return []
    energy = _short_time_rms(y, frame, hop)
    floor = float(np.percentile(energy, 20))
    typical = float(np.percentile(energy, 75))
    noise_threshold = floor * 2.5 if floor < typical * 0.25 else 0.0
    active = energy > max(0.008, noise_threshold, typical * 0.035)

    # Bridge brief unvoiced gaps, then discard clicks and other very short bursts.
    max_gap = max(1, int(0.18 * sr / hop))
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    merged: list[list[int]] = []
    for start, end in zip(starts, ends):
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    min_frames = max(1, int(0.10 * sr / hop))
    return [(start * hop / sr, min(len(y) / sr, (end * hop + frame) / sr))
            for start, end in merged if end - start >= min_frames]


# ---------------------------------------------------------------- speakers
@lru_cache(maxsize=1)
def _speaker_encoder():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(os.path.expanduser("~"), ".cache", "classlens", "spkrec-ecapa-voxceleb"),
        run_opts={"device": device},
    )


def assign_speakers(segments: list[Segment], y: np.ndarray, sr: int,
                    progress: Callable[[str], None] | None = None):
    """Cluster local ECAPA speaker embeddings; the longest-speaking cluster is the teacher."""
    info = {"method": "ecapa", "confident": False, "silhouette": None, "speakers": 1}
    for s in segments:
        s.speaker = "teacher"
    if len(segments) < 8:
        return segments, info
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        info["error"] = "SpeechBrain ECAPA model unavailable while Hugging Face offline mode is enabled."
        return segments, info
    try:
        import librosa
        import torch
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        if progress:
            progress("Loading local ECAPA speaker model; first use may download model weights")
        clips: list[tuple[int, np.ndarray, float, float]] = []
        encoder = _speaker_encoder()
        for index, s in enumerate(segments):
            a = int(s.start * sr)
            b = int(min(s.end, s.start + 8) * sr)
            clip = y[a:b]
            if len(clip) < int(0.65 * sr):
                continue
            pitch = librosa.yin(clip, fmin=65, fmax=500, sr=sr)
            median_pitch = float(np.median(pitch)) if len(pitch) else 0.0
            clips.append((index, np.asarray(clip, dtype=np.float32), median_pitch, s.end - s.start))
        valid: list[tuple[int, np.ndarray, float, float]] = []
        for offset in range(0, len(clips), 16):
            batch = clips[offset:offset + 16]
            if progress:
                progress(f"Creating speaker embeddings: {min(offset + len(batch), len(clips))}/{len(clips)}")
            max_length = max(len(clip) for _, clip, _, _ in batch)
            waveforms = np.zeros((len(batch), max_length), dtype=np.float32)
            lengths = np.empty(len(batch), dtype=np.float32)
            for row, (_, clip, _, _) in enumerate(batch):
                waveforms[row, :len(clip)] = clip
                lengths[row] = len(clip) / max_length
            with torch.no_grad():
                encoded = encoder.encode_batch(
                    torch.from_numpy(waveforms),
                    wav_lens=torch.from_numpy(lengths),
                ).detach().cpu().numpy()
            valid.extend(
                (batch[row][0], embedding.reshape(-1), batch[row][2], batch[row][3])
                for row, embedding in enumerate(encoded)
            )
        max_speakers = min(5, len(valid) // 3)
        if max_speakers < 2:
            info["speakers"] = 1
            return segments, info

        embeddings = np.vstack([v[1] for v in valid])
        pitch = np.log(np.maximum([v[2] for v in valid], 1.0))
        pitch_std = float(np.std(pitch))
        if pitch_std > 1e-6:
            pitch = ((pitch - np.mean(pitch)) / pitch_std * 1.5).reshape(-1, 1)
            embeddings = np.column_stack((embeddings, pitch))

        best = None
        for n_speakers in range(2, max_speakers + 1):
            labels = AgglomerativeClustering(n_clusters=n_speakers, linkage="average", metric="cosine").fit_predict(embeddings)
            counts = np.bincount(labels)
            if counts.min() < 3:
                continue
            silhouette = float(silhouette_score(embeddings, labels, metric="cosine"))
            if best is None or silhouette > best[0]:
                best = (silhouette, labels, n_speakers)

        if best is None:
            return segments, info
        sil, labels, n_speakers = best
        info.update(silhouette=round(sil, 3), speakers=n_speakers)
        if sil < 0.12:
            return segments, info

        durations = np.zeros(n_speakers)
        for (_, _, _, duration), label in zip(valid, labels):
            durations[label] += duration
        teacher_cluster = int(np.argmax(durations))
        for (index, _, _, _), label in zip(valid, labels):
            segments[index].speaker = "teacher" if label == teacher_cluster else "student"
        info["confident"] = True
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    return segments, info


# ---------------------------------------------------------------- helpers
def _turns(segs: list[Segment], gap: float = 3.0):
    turns = []
    for s in segs:
        confidence = _confidence_weight(s)
        if turns and turns[-1]["speaker"] == s.speaker and s.start - turns[-1]["end"] <= gap:
            turns[-1]["end"] = s.end
            turns[-1]["text"] += " " + s.text
            turns[-1]["confidence_sum"] += confidence
            turns[-1]["confidence_count"] += 1
            turns[-1]["weighted_words"] += words(s.text) * confidence
        else:
            turns.append({
                "speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text,
                "confidence_sum": confidence, "confidence_count": 1,
                "weighted_words": words(s.text) * confidence,
            })
    for t in turns:
        t["words"] = words(t["text"])
        t["confidence"] = t["confidence_sum"] / t["confidence_count"]
    return turns


def _audio_features(y: np.ndarray, sr: int, segs: list[Segment]) -> dict:
    f = {}
    rms = _short_time_rms(y, 1024, 512)
    if len(rms):
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
    try:
        import librosa
    except ImportError:
        f["pitch_estimation_available"] = False
        return f
    f["pitch_estimation_available"] = True
    # teacher pitch / intonation range (semitones, p90-p10) on a sample of teacher segments
    teacher = [s for s in segs if s.speaker == "teacher" and s.end - s.start > 1.0]
    indices = np.linspace(0, len(teacher) - 1, min(40, len(teacher))).astype(int) if teacher else []
    pitch_values, semitones = [], []
    for index in indices:
        s = teacher[index]
        clip = y[int(s.start * sr):int(min(s.end, s.start + 6) * sr)]
        if len(clip) < 2048:
            continue
        f0 = librosa.yin(clip, fmin=65, fmax=500, sr=sr, frame_length=1024)
        f0 = f0[np.isfinite(f0) & (f0 >= 65) & (f0 <= 500)]
        if len(f0):
            pitch_values.append(f0)
            semitones.append(12 * np.log2(f0 / np.median(f0)))
    if pitch_values:
        f["teacher_median_pitch_hz"] = float(np.median(np.concatenate(pitch_values)))
        allv = np.concatenate(semitones)
        f["teacher_pitch_range_st"] = float(np.percentile(allv, 90) - np.percentile(allv, 10))
    return f


def _is_rising_pitch(y: np.ndarray, sr: int, segment: Segment) -> bool:
    try:
        import librosa
    except ImportError:
        return False
    end = int(segment.end * sr)
    clip = y[max(0, end - sr):end]
    if len(clip) < int(0.4 * sr):
        return False
    f0 = librosa.yin(clip, fmin=65, fmax=500, sr=sr)
    if len(f0) < 8:
        return False
    split = len(f0) // 2
    return bool(12 * np.log2(np.median(f0[split:]) / np.median(f0[:split])) >= 1.2)


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
    f["screen_ocr_available"] = False
    if shutil.which("tesseract"):
        try:
            import pytesseract

            texts = []
            for _, frame in frames:
                data = pytesseract.image_to_data(
                    np.clip(frame, 0, 255).astype(np.uint8),
                    config="--psm 11",
                    output_type=pytesseract.Output.DICT,
                )
                texts.append([text.strip().lower() for text, conf in zip(data["text"], data["conf"])
                              if text.strip() and float(conf) >= 50])
            f["screen_ocr_available"] = True
            f["screen_text_words_per_frame"] = float(np.mean([len(text) for text in texts]))
            f["screen_text_change_ratio"] = float(
                np.mean([texts[i] != texts[i - 1] for i in range(1, len(texts))])
            )
        except (OSError, RuntimeError, ValueError) as exc:
            f["screen_ocr_error"] = str(exc)[:200]
    return f


# ---------------------------------------------------------------- main
def compute_features(segs: list[Segment], y: np.ndarray, sr: int, frames=None, frame_step: float = 10.0,
                     diar_confident: bool = True, vad_audio: np.ndarray | None = None) -> dict:
    dur = len(y) / sr
    minutes = dur / 60
    f: dict = {"duration_s": dur, "n_segments": len(segs), "diarization_confident": diar_confident}
    voice = _voice_activity_intervals(vad_audio if vad_audio is not None else y, sr)
    f["vad_speech_seconds"] = sum(end - start for start, end in voice)
    T = [s for s in segs if s.speaker == "teacher"]
    S = [s for s in segs if s.speaker == "student"]
    tdur = sum((s.end - s.start) * _confidence_weight(s) for s in T)
    sdur = sum((s.end - s.start) * _confidence_weight(s) for s in S)
    tw = sum(words(s.text) * _confidence_weight(s) for s in T)
    sw = sum(words(s.text) * _confidence_weight(s) for s in S)
    f.update(teacher_words=tw, student_words=sw, speech_ratio=f["vad_speech_seconds"] / dur if dur else 0,
             student_share=sdur / (tdur + sdur) if tdur + sdur else 0,
             teacher_wpm=tw / (tdur / 60) if tdur > 5 else 0.0)

    turns = _turns(segs)
    t_turns = [t for t in turns if t["speaker"] == "teacher" and t["confidence"] >= 0.35]
    s_turns = [t for t in turns if t["speaker"] == "student"]
    f["student_turns"] = len(s_turns)
    f["student_turns_per_min"] = sum(t["confidence"] for t in s_turns) / minutes if minutes else 0
    reliable_student_turns = [t for t in s_turns if t["confidence"] >= 0.35]
    f["avg_student_turn_words"] = float(np.mean([t["weighted_words"] for t in reliable_student_turns])) if reliable_student_turns else 0.0
    f["one_word_answer_share"] = float(np.mean([t["words"] <= 2 for t in reliable_student_turns])) if reliable_student_turns else 0.0
    ev: dict = {}
    if t_turns:
        lm = max(t_turns, key=lambda t: t["end"] - t["start"])
        f["longest_teacher_monologue_s"] = lm["end"] - lm["start"]
        ev["monologue"] = (lm["start"], lm["end"] - lm["start"], lm["text"][:140])
    f["teacher_turns_over_90s"] = sum(1 for t in t_turns if t["end"] - t["start"] > 90)

    # questions and responses
    q_n, open_n, answered, own_answered, waits, qs = 0, 0, 0, 0, [], []
    f["rising_pitch_questions"] = 0
    reliable_teacher = [(i, s) for i, s in enumerate(segs)
                        if s.speaker == "teacher" and _confidence_weight(s) >= 0.35]
    pitch_candidates = [(i, s) for i, s in reliable_teacher if "?" not in s.text]
    if len(pitch_candidates) > 60:
        selected = np.linspace(0, len(pitch_candidates) - 1, 60).astype(int)
        pitch_candidates = [pitch_candidates[index] for index in selected]
    rising_indices = {i for i, segment in pitch_candidates if _is_rising_pitch(y, sr, segment)}
    for i, s in reliable_teacher:
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", s.text) if x.strip().endswith("?")]
        rising = i in rising_indices
        if rising:
            f["rising_pitch_questions"] += 1
        if not sentences and not rising:
            continue
        q_n += len(sentences) if sentences else 1
        open_n += sum(bool(OPEN_Q.match(x)) for x in sentences)
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        outcome = "unanswered"
        wait = None
        if nxt is not None:
            if nxt.speaker == "student" and _confidence_weight(nxt) >= 0.35 and nxt.start - s.end < 15:
                speech_after_question = next(
                    (start for start, _ in voice if start >= s.end - 0.1 and start <= nxt.end),
                    None,
                )
                if speech_after_question is not None:
                    answered += 1
                    wait = max(0.0, speech_after_question - s.end)
                    waits.append(wait)
                    outcome = "answered"
            elif nxt.speaker == "teacher" and nxt.start - s.end < 3:
                own_answered += 1
                outcome = "teacher continued"
        qs.append((s.start, (sentences[-1] if sentences else s.text)[:120], outcome, wait))
    f["questions"] = q_n
    f["questions_per_min"] = q_n / minutes if minutes else 0
    f["open_question_share"] = open_n / q_n if q_n else 0.0
    f["question_response_rate"] = answered / len(qs) if qs else 0.0
    f["teacher_answers_own_question_share"] = own_answered / len(qs) if qs else 0.0
    f["median_wait_s"] = float(np.median(waits)) if waits else None
    ev["questions"] = qs[:6]

    # Long silences come from audio activity, not Whisper's segment boundaries.
    gaps = [(voice[i][1], voice[i + 1][0] - voice[i][1]) for i in range(len(voice) - 1)]
    long = [(a, gap) for a, gap in gaps if gap > 8.0]
    f["dead_air_s"] = float(sum(g for _, g in long))
    f["dead_air_ratio"] = f["dead_air_s"] / dur if dur else 0
    f["dead_air_events"] = len(long)
    if long:
        ev["longest_gap"] = max(long, key=lambda x: x[1])

    # language cues (teacher)
    trusted_teacher = [s for s in T if _confidence_weight(s) >= 0.35]
    praise = [(s.start, s.text) for s in trusted_teacher if PRAISE.search(s.text)]
    f["praise_per_min"] = sum(len(PRAISE.findall(s.text)) * _confidence_weight(s) for s in trusted_teacher) / minutes if minutes else 0
    f["correction_per_min"] = sum(len(CORRECT.findall(s.text)) * _confidence_weight(s) for s in trusted_teacher) / minutes if minutes else 0
    f["filler_per_100_words"] = 100 * sum(len(FILLER.findall(s.text)) * _confidence_weight(s) for s in trusted_teacher) / max(tw, 1)
    f["engagement_prompts_per_min"] = sum(len(PROMPT.findall(s.text)) * _confidence_weight(s) for s in trusted_teacher) / minutes if minutes else 0
    tech = [(s.start, s.text) for s in segs if _confidence_weight(s) >= 0.35 and TECH.search(s.text)]
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


def apply_interaction_tags(features: dict, tags: dict | None, segments: list[Segment]) -> None:
    """Use only verbatim LLM tags that can be matched back to the local transcript."""
    if not isinstance(tags, dict):
        return
    transcript = [
        "".join(re.findall(r"\w+", s.text.casefold()))
        for s in segments if s.speaker == "teacher" and _confidence_weight(s) >= 0.35
    ]
    valid: dict[str, list[dict[str, str]]] = {"questions": [], "praise": []}
    for kind in valid:
        for item in tags.get(kind, [])[:30]:
            if not isinstance(item, dict):
                continue
            quote = item.get("quote")
            timestamp = item.get("timestamp")
            if not isinstance(quote, str) or not quote.strip():
                continue
            normalized = "".join(re.findall(r"\w+", quote.casefold()))
            if len(normalized) < 4 or not any(normalized in text for text in transcript):
                continue
            valid[kind].append({"timestamp": str(timestamp or ""), "quote": quote[:120]})
    minutes = max(float(features.get("duration_s", 0)) / 60, 1e-6)
    if valid["questions"]:
        features["questions"] = max(features.get("questions", 0), len(valid["questions"]))
        features["questions_per_min"] = max(features.get("questions_per_min", 0), len(valid["questions"]) / minutes)
        features.setdefault("_evidence", {})["llm_questions"] = valid["questions"]
    if valid["praise"]:
        features["praise_per_min"] = max(features.get("praise_per_min", 0), len(valid["praise"]) / minutes)
        features.setdefault("_evidence", {})["llm_praise"] = valid["praise"]
