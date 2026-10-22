# Pedagogical-Quality-Interaction-Analytics (ClassLens)

**ClassLens — Classroom Interaction & Pedagogy Analytics**

ClassLens is a local-first tool for reviewing recorded online lessons. Upload an audio/video recording or provide a supported media link; the app returns a **Poor / Average / Good / Excellent** rubric result, measured interaction signals, transcript, and timestamped evidence. It consists of a FastAPI service and a browser-based dashboard.

> **Important:** ClassLens provides automated, approximate indicators—not a definitive or high-stakes evaluation of a teacher, student, or lesson. Review the evidence and limitations before using results.

## Project description

The hosted dashboard follows a deliberately small, deterministic path: it reduces steady background noise, transcribes locally with Whisper `tiny`, measures audio and transcript signals, and scores the result without sending data to an external AI service. It does not process video frames or separate speakers, so participation estimates are unavailable. The analysis modules retain optional capabilities for local experimentation, but they are not enabled by the web service.

For the full product description, see [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md). For click-by-click instructions and troubleshooting, see [USER_GUIDE.md](USER_GUIDE.md).

## Features

- Upload local audio/video or ingest supported Google Drive, direct HTTP, and yt-dlp-supported media links.
- Monitor an asynchronous analysis job and view its result in the dashboard.
- Inspect the transcript, teacher/student estimate, rubric dimensions, score, and timestamped observations.
- Split word-timestamped transcript segments at pauses, retain per-segment Whisper confidence, and use audio activity for silence and wait-time estimates.
- Run the hosted dashboard with a single CPU-friendly Whisper model and deterministic scoring.
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

The repository includes a Render Blueprint (`render.yaml`) and Docker image definition. In Render, create or sync a Blueprint from this repository and select the `main` branch. The service installs FFmpeg and its development libraries for PyAV. The Docker build verifies PyAV, FFmpeg, and NumPy pitch estimation; the pitch estimator avoids runtime Numba cache failures in librosa. The service checks `/health` and uses Whisper `tiny`, deterministic scoring, audio-only analysis, and disabled speaker separation to limit CPU and memory use. A 2 GB persistent disk at `/var/data` caches Whisper weights across deploys; persistent storage can add Render charges and causes brief downtime during deploys. If the existing service is not managed by this Blueprint, add the disk in Render's service settings with that mount path. The Blueprint generates an `APP_PASSWORD`; find its value in the service's environment settings and enter it in the dashboard before submitting an assessment. The password also protects report and job endpoints.

The first assessment downloads speech-model weights and requires outbound network access. Jobs and the latest report are held in process memory and are lost after a service restart; the dashboard retries temporary gateway errors, but a lost job must be resubmitted. Uploaded recordings are processed from temporary storage and removed after analysis. The dashboard accepts recordings up to 15 minutes, uses one Whisper `tiny` model, and does not enable frame extraction, speaker separation, or external AI review. Combined-voice mode cannot estimate teacher/student participation, and the report calls this out. Do not use this public service for recordings unless you have appropriate consent and access controls. Tailwind styles are precompiled and served locally rather than loaded from the development CDN. After changing dashboard utility classes, run `npm install` and `npm run build:css` to refresh `web/styles.css`.

The hosted web service uses deterministic scoring only. It does not accept provider API keys or send transcripts to external AI services. To protect assessment routes, configure `APP_PASSWORD` in the deployment environment and enter it in the dashboard; never commit passwords or `.env` files.

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

`POST /assess` accepts multipart form fields `file`, `url`, or `sample`, with optional `max_minutes` (1–15) and `token`. It always uses Whisper `tiny`, deterministic scoring, and audio-only analysis. It returns a job ID before analysis starts; poll `GET /jobs/{id}` for progress. FastAPI runs analysis as a post-response background task. Jobs are held in process memory; restarting the service clears jobs and the latest report.

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
