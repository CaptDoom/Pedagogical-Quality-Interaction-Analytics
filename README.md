# Pedagogical-Quality-Interaction-Analytics (ClassLens)

**ClassLens — Classroom Interaction & Pedagogy Analytics**

ClassLens is a local-first tool for reviewing recorded online lessons. Upload an audio/video recording or provide a supported media link; the app returns a **Poor / Average / Good / Excellent** rubric result, measured interaction signals, transcript, and timestamped evidence. It consists of a FastAPI service and a browser-based dashboard.

> **Important:** ClassLens provides automated, approximate indicators—not a definitive or high-stakes evaluation of a teacher, student, or lesson. Review the evidence and limitations before using results.

## Project description

The pipeline extracts audio, transcribes speech locally with faster-whisper, estimates teacher/student turns, measures interaction and delivery signals, and scores those signals using a configurable rubric. Optional LLM review can add qualitative feedback. When that review is enabled, **transcript text and aggregate metrics** are sent to the selected provider; audio and video frames are not sent.

For the full product description, see [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md). For click-by-click instructions and troubleshooting, see [USER_GUIDE.md](USER_GUIDE.md).

## Features

- Upload local audio/video or ingest supported Google Drive, direct HTTP, and yt-dlp-supported media links.
- Monitor an asynchronous analysis job and view its result in the dashboard.
- Inspect the transcript, teacher/student estimate, rubric dimensions, score, and timestamped observations.
- Optionally sample video frames for screen-change signals.
- Choose deterministic scoring, a local Ollama backend, Gemini, or an OpenAI-compatible endpoint.
- Run the FastAPI interactive API documentation at `/api`.

## Quick start

Requirements: Python 3.10 or newer, FFmpeg (including `ffprobe`) on `PATH`, and the packages in `requirements.txt`.

### Windows PowerShell

```powershell
cd path\to\classlens
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python server.py
```

If PowerShell blocks environment activation, either adjust the execution policy for the current process or call `.venv\Scripts\python.exe` directly. Install FFmpeg separately and confirm both `ffmpeg` and `ffprobe` resolve from the terminal.

### macOS / Linux

```bash
cd path/to/classlens
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python server.py
```

Install FFmpeg with the operating system's package manager first. The service listens on `http://127.0.0.1:8000`; open that address in a browser. Set `PORT` to change the port. `python server.py` binds to loopback only; do not expose this development server directly to the public internet.

## Configure optional LLM review

Create a `.env` file in the project directory (it is ignored by Git) and add only the settings for your chosen provider:

```dotenv
# Optional: Gemini
GEMINI_API_KEY=your-key
# Optional: OpenAI-compatible service
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.openai.com/v1
# Optional: local Ollama
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=qwen2.5:7b
# Optional: require a password on assessment requests
# APP_PASSWORD=choose-a-long-random-password
```

The UI also accepts a per-request provider key and, when configured, the application password. Never commit `.env` or paste secrets into issue reports. Select **None (Deterministic only)** to avoid sending transcript data to an external LLM. See [USER_GUIDE.md](USER_GUIDE.md) for data handling details.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Dashboard |
| `GET` | `/api` | Interactive OpenAPI documentation |
| `GET` | `/api/samples` | List supported local demo media |
| `GET` | `/api/latest` | Latest completed report in this process |
| `POST` | `/assess` | Submit a file, URL, or sample; returns a job ID |
| `GET` | `/jobs/{id}` | Poll job state, progress messages, and report |
| `GET` | `/api/media?path=...` | Stream a project-local media asset |

`POST /assess` accepts multipart form fields such as `file`, `url`, `sample`, `whisper_model`, `use_frames`, `max_minutes`, `llm_backend`, `api_key`, and `token`. Assessment jobs are held in process memory; restarting the service clears jobs and the latest report.

## Test

Run from this directory using the same Python environment as the app:

```bash
python -m pytest tests -q
```

The suite includes rubric and feature checks, ingestion and download tests, LLM failure behavior, API validation, media path containment, and asynchronous job lifecycle checks.

## Calibration and known limitations

The current rubric was calibrated against only three labelled recordings. That is not enough data to claim general accuracy or validate borderline results. Speaker assignment uses voice-timbre clustering and can be unreliable with overlapping speech, multiple children, or similar voices. Whisper transcription can be weak for young speakers and noisy audio; language cues are primarily English. Wait-time measurement is currently limited by segment timing. Screen analysis detects visual changes, not the instructional value of a slide or screen.

To evaluate a larger labelled dataset, run `python calibrate.py samples --model small --frames` (optionally add `--holdout N`). Keep labelled recordings and generated calibration output out of public commits unless you have permission to share them.

## Privacy and repository contents

Processing is local by default. Temporary uploaded copies, extracted audio, and sampled frames are removed after analysis. If an external LLM is selected, transcript text and aggregated measurements are transmitted to that provider. Obtain the necessary consent before processing classroom recordings, especially recordings involving minors.

The repository intentionally ignores `.env`, virtual environments, model/tool downloads, caches, and sample media/recordings. Install dependencies locally and supply your own authorized test recordings. Do not expose this service publicly without adding production-grade authentication, deployment hardening, and a privacy review.
