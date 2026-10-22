# Pedagogical-Quality-Interaction-Analytics (ClassLens)

**ClassLens — Classroom Interaction & Pedagogy Analytics**

ClassLens is a local-first tool for reviewing recorded online lessons. Upload an audio/video recording or provide a supported media link; the app returns a **Poor / Average / Good / Excellent** rubric result, measured interaction signals, transcript, and timestamped evidence. It consists of a FastAPI service and a browser-based dashboard.

> **Important:** ClassLens provides automated, approximate indicators—not a definitive or high-stakes evaluation of a teacher, student, or lesson. Review the evidence and limitations before using results.

## Project description

The pipeline reduces steady background noise, transcribes speech locally with faster-whisper, estimates teacher/student turns, measures interaction and delivery signals, and scores those signals using a configurable rubric. Optional LLM review can add qualitative feedback. When that review is enabled, **transcript text and aggregate metrics** are sent to the selected provider; audio and video frames are not sent. Sampled-frame OCR runs locally with Tesseract and sends no images or extracted text to an external service.

For the full product description, see [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md). For click-by-click instructions and troubleshooting, see [USER_GUIDE.md](USER_GUIDE.md).

## Features

- Upload local audio/video or ingest supported Google Drive, direct HTTP, and yt-dlp-supported media links.
- Monitor an asynchronous analysis job and view its result in the dashboard.
- Inspect the transcript, teacher/student estimate, rubric dimensions, score, and timestamped observations.
- Optionally sample video frames for low-weight screen-change signals and local OCR text-density/change measurements.
- Split word-timestamped transcript segments at pauses, retain per-segment Whisper confidence, and use audio activity for silence and wait-time estimates.
- Estimate speaker embeddings locally with SpeechBrain ECAPA; install its model weights on first use. If embedding extraction or clustering is unreliable, speaker separation remains explicitly unconfident and is scored neutrally.
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

Install FFmpeg with the operating system's package manager first. The service listens on `http://127.0.0.1:8000`; open that address in a browser. Set `PORT` to change the port. `python server.py` binds to loopback by default; the Render configuration explicitly binds to `0.0.0.0`.

### Deploy to Render

The repository includes a Render Blueprint (`render.yaml`) and Docker image definition. In Render, create or sync a Blueprint from this repository and select the `main` branch. The service uses the Standard plan, installs FFmpeg, its development headers and `pkg-config` (so PyAV can build if pip has no compatible wheel), and Tesseract. The Docker build verifies that PyAV imports, FFmpeg is available, and NumPy pitch estimation works as the non-root application user. Pitch cues use an internal NumPy YIN estimator, avoiding runtime Numba cache failures in librosa. It checks `/health` and defaults to the CPU-faster Whisper `base` model, deterministic review (no external LLM request), and speaker diarization disabled. A 2 GB persistent disk at `/var/data` caches Whisper and optional SpeechBrain weights across deploys; persistent storage can add Render charges and causes brief downtime during deploys. If the existing service is not managed by this Blueprint, add the disk in Render's service settings with that mount path. The Blueprint generates an `APP_PASSWORD`; find its value in the service's environment settings and enter it in the dashboard before submitting an assessment. The password also protects report and job endpoints.

The first assessment downloads speech-model weights and requires outbound network access. The model cache and in-memory jobs/reports are not persistent across a new deployment or instance replacement. Uploaded recordings are processed from temporary storage and removed after analysis. Do not use this public service for recordings unless you have appropriate consent and access controls. For local and hosted use, the dashboard defaults to Whisper `base`, a 15-minute analysis limit, visual frame analysis off, and speaker diarization off to avoid unexpectedly long CPU jobs. Speaker diarization can be enabled per run; frame sampling/OCR is also optional in Pipeline Settings. Combined-voice mode does not estimate teacher/student participation, and the report calls this out. Choose the slower `small` or `medium` model, a longer duration, or diarization when the accuracy tradeoff is acceptable. Tailwind styles are precompiled and served locally rather than loaded from the development CDN. After changing dashboard utility classes, run `npm install` and `npm run build:css` to refresh `web/styles.css`.

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
| `GET` | `/health` | Render health check |
| `GET` | `/api/config` | Configured default analysis options |
| `GET` | `/api/samples` | List supported local demo media |
| `GET` | `/api/latest` | Latest completed report in this process |
| `POST` | `/assess` | Submit a file, URL, or sample; returns a job ID |
| `GET` | `/jobs/{id}` | Poll job state, progress messages, and report |
| `GET` | `/api/media?path=...` | Stream a project-local media asset |

`POST /assess` accepts multipart form fields such as `file`, `url`, `sample`, `whisper_model`, `use_frames`, `max_minutes`, `llm_backend`, `api_key`, and `token`. It returns a job ID before analysis starts; poll `GET /jobs/{id}` for progress. FastAPI runs analysis as a post-response background task. Jobs are held in process memory; restarting the service clears jobs and the latest report.

## Test

Run from this directory using the same Python environment as the app:

```bash
python -m pytest tests -q
```

The suite includes rubric and feature checks, ingestion and download tests, LLM failure behavior, API validation, media path containment, and asynchronous job lifecycle checks.

## Calibration and known limitations

Run `python calibrate.py samples --model small --frames` to re-analyse every labeled recording, perform leave-one-recording-out (LOO) validation, print exact and within-one-level accuracy, a four-level confusion matrix, pairwise ranking concordance, fitted cutoffs/weights, and signal monotonicity. The script stores fresh feature results in `.cache/` and writes LOO predictions to `calibration.csv`. Add `--write-config` to persist full-dataset fitted cutoffs and weights after reviewing the report. The `quality issue` folder is mapped to Average per the dataset labeling instruction. The available labeled recordings still contain no Poor example; the Poor level and its adjacent cutoff therefore remain untested. Small, imbalanced samples still cannot establish general accuracy.

Latest cached run (2026-10-22; Whisper `small`, without frame sampling): seven recordings (3 Average, 2 Good, 2 Excellent) produced **42.9% exact accuracy**, **71.4% within-one-level accuracy**, and **75.0% pairwise ranking concordance** across 16 differently labeled pairs. Confusion matrix (true rows, predicted columns):

| True \ Predicted | Poor | Average | Good | Excellent |
| --- | ---: | ---: | ---: | ---: |
| Poor | 0 | 0 | 0 | 0 |
| Average | 0 | 2 | 0 | 1 |
| Good | 0 | 2 | 0 | 0 |
| Excellent | 0 | 1 | 0 | 1 |

The full-data diagnostic fit returned cutoffs Poor `2.60`, Average `3.471`, and Good `3.471`; the equal Average/Good cutoffs do not establish a useful Good band, and these fitted values were **not** written to `config.json`. Student-word counts were zero in all seven cached analyses, so this run does not validate student-participation signals. Interpret all results cautiously: the dataset is small and imbalanced, contains no Poor example, and has a known speaker-assignment limitation.

Speaker embeddings and pitch cues improve turn assignment but can still fail with overlapping speech, several similar children, or short turns; inspect the diarization confidence before interpreting participation. Whisper uses word timestamps, reduces background noise before transcription, and pins a detected language when its confidence is high. Low-confidence text contributes less to lexical measurements and is marked in the transcript. Questions can also be estimated from rising pitch; optional LLM tags can recognize praise/questions across languages when an approved backend is selected. Silence and wait time use audio activity, not just ASR boundaries.

Frame OCR requires both the `pytesseract` Python package and a local Tesseract executable on `PATH`. Install them locally to enable text-density and text-change measurements. Screen signals have deliberately low rubric weight until calibration demonstrates that they separate quality levels. Frame sampling and OCR stay local; do not send images containing children to external services.

Keep labeled recordings and generated calibration output out of public commits unless you have permission to share them.

## Privacy and repository contents

Processing is local by default. Temporary uploaded copies, extracted audio, and sampled frames are removed after analysis. If an external LLM is selected, transcript text and aggregated measurements are transmitted to that provider. Obtain the necessary consent before processing classroom recordings, especially recordings involving minors.

The repository intentionally ignores `.env`, virtual environments, model/tool downloads, caches, and sample media/recordings. Install dependencies locally and supply your own authorized test recordings. SpeechBrain's ECAPA model and Whisper models may download model weights on first use; this inference runs locally. Do not expose this service publicly without adding production-grade authentication, deployment hardening, and a privacy review.
