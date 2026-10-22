"""Optional LLM layer. Qualitative judgement (clarity, rapport, content accuracy) plus timestamped evidence.

Recordings involve children: only enable a backend the organiser has approved. 'ollama' keeps everything local.
Only the TRANSCRIPT and aggregate metrics are sent; audio and video frames never are.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .types import Segment, fmt

SYSTEM = (
    "You are an expert reviewer of online classes for children. You judge teaching quality from a transcript "
    "(T = teacher, S = student; speaker labels may be imperfect) and measured metrics. Be specific: every point "
    "must cite a timestamp [mm:ss] and a short verbatim quote (under 15 words) from the transcript. "
    "Ignore transcript stretches with confidence below 0.4. Tag teacher questions and praise in any language, "
    "but only when explicitly present in the transcript; never translate or invent a quote. "
    "Do not judge on length or volume alone. Return ONLY JSON."
)

SCHEMA = (
    '{"clarity": 1-5, "rapport": 1-5, "content_accuracy": 1-5, "summary": "2 sentences", '
    '"strengths": [{"timestamp": "mm:ss", "observation": "...", "quote": "..."}], '
    '"weaknesses": [{"timestamp": "mm:ss", "observation": "...", "quote": "..."}], '
    '"interaction_tags": {"questions": [{"timestamp": "mm:ss", "quote": "..."}], '
    '"praise": [{"timestamp": "mm:ss", "quote": "..."}]}}'
)


def render_transcript(segs: list[Segment], budget: int = 14000, block: int = 12) -> str:
    lines = [
        f"[{fmt(s.start)}] {'T' if s.speaker == 'teacher' else 'S'}"
        f"{f' (confidence {s.confidence:.2f})' if s.confidence is not None else ''}: {s.text}"
        for s in segs
    ]
    if sum(len(l) + 1 for l in lines) <= budget:
        return "\n".join(lines)
    blocks = [lines[i:i + block] for i in range(0, len(lines), block)]
    per = max(1, budget // max(1, len(blocks[0]) * 70))
    step = max(1, len(blocks) // per)
    chosen = blocks[::step]
    return "\n...\n".join("\n".join(b) for b in chosen)[:budget]


def build_prompt(segs, features: dict) -> str:
    keys = ["duration_s", "student_share", "student_turns", "questions_per_min", "open_question_share",
            "median_wait_s", "teacher_wpm", "praise_per_min", "dead_air_ratio", "longest_teacher_monologue_s"]
    m = {k: (round(features[k], 2) if isinstance(features.get(k), float) else features.get(k)) for k in keys}
    return (f"Metrics: {json.dumps(m)}\n\nTranscript (sampled if long):\n{render_transcript(segs)}\n\n"
            f"Rate the class and return JSON with this shape:\n{SCHEMA}")


def _post(url, payload, headers, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        raise RuntimeError(f"backend returned HTTP {e.code} {e.reason}: {body}") from e


def _parse(text: str) -> dict:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("LLM returned no JSON")
    return json.loads(text[a:b + 1])


def _call_gemini(key: str, prompt: str) -> dict:
    models_to_try = [
        os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
    ]
    seen = set()
    models = [m for m in models_to_try if not (m in seen or seen.add(m))]
    last_err = None
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": SYSTEM + "\n\n" + prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        try:
            out = _post(url, payload, {"x-goog-api-key": key})
            text = out["candidates"][0]["content"]["parts"][0]["text"]
            return _parse(text)
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Gemini request failed.")


def run(backend: str, segs, features, api_key: str | None = None) -> dict | None:
    if backend in (None, "", "none"):
        return None
    prompt = build_prompt(segs, features)
    if backend == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        out = _post(f"{host}/api/chat", {
            "model": os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"), "stream": False, "format": "json",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}, {})
        return _parse(out["message"]["content"])
    if backend == "anthropic":
        key = (api_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        out = _post("https://api.anthropic.com/v1/messages", {
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"), "max_tokens": 1500, "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}]},
            {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return _parse(out["content"][0]["text"])
    if backend in ("gemini", "google"):
        key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("API key is not set. Please provide a Gemini or OpenAI key in .env or the UI.")
        if key.startswith("sk-"):
            base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            out = _post(f"{base}/chat/completions", {
                "model": model, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]},
                {"Authorization": f"Bearer {key}"})
            return _parse(out["choices"][0]["message"]["content"])
        return _call_gemini(key, prompt)
    if backend == "openai-compatible":
        key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (put it in .env or the environment).")
        if not key.startswith("sk-"):
            return _call_gemini(key, prompt)
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        out = _post(f"{base}/chat/completions", {
            "model": model, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]},
            {"Authorization": f"Bearer {key}"})
        return _parse(out["choices"][0]["message"]["content"])
    raise ValueError(f"Unknown backend {backend}")
