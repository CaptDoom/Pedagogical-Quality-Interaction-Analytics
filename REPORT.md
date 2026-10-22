# ClassLens — Build and Verification Report

**Updated:** 22 October 2026

**Project:** Pedagogical-Quality-Interaction-Analytics (ClassLens)

## Scope

This report records the current application structure and checks performed while preparing the repository. It avoids machine-specific paths and secrets. For installation and usage instructions, see [README.md](README.md) and [USER_GUIDE.md](USER_GUIDE.md).

## Application

ClassLens is a FastAPI service with a browser-based dashboard. Users submit a recording for asynchronous analysis and receive a rubric score, transcript, measured signals, and timestamped observations. Media processing and transcription are local by default. Optional external LLM providers receive transcript text and aggregate metrics only.

## Verification performed

- The project test suite completed successfully: **18 passed** using the project virtual environment.
- Automated checks cover scoring/features, ingestion and HTTP downloads, graceful LLM failure, HTTP routes and request validation, media path containment, password enforcement, and asynchronous assessment job lifecycle.
- API smoke checks returned the expected responses for the dashboard, samples, latest report, media traversal rejection, and duration validation.
- The dashboard was opened in a browser and the assessment dialog and sample list were checked.

The automated pipeline tests use synthetic or stubbed analysis inputs where noted in the tests. This verification does **not** establish classification accuracy on unseen real-world recordings, validate live external provider credentials, or verify every third-party link provider with a real account/link.

## Files and components

| Path | Responsibility |
| --- | --- |
| `server.py` | FastAPI routes, assessment job lifecycle, password and input validation |
| `web/index.html` | Dashboard, ingestion dialog, progress polling, report display |
| `assessor/pipeline.py` | Media analysis orchestration and report construction |
| `assessor/media.py` | Media ingestion, validation, downloads, audio extraction, optional frame sampling |
| `assessor/transcribe.py` | Local faster-whisper transcription |
| `assessor/features.py` | Speaker estimation and interaction/audio/video signal extraction |
| `assessor/scoring.py` | Weighted rubric and evidence generation |
| `assessor/llm.py` | Optional qualitative review provider integrations |
| `config.json` | Rubric thresholds and weights |
| `tests/` | Automated unit and API tests |

## Data and repository safety

The repository ignores `.env`, local virtual environments, downloaded tools, caches, calibration output, and recordings/sample media. Keep credentials and recordings out of source control. Ensure that the ignore rules are maintained when adding future media formats. Review privacy and consent requirements before processing classroom recordings, especially recordings involving minors.

## Known limitations

- Calibration currently reflects a very small labeled dataset and should not be generalized as a validated accuracy claim.
- Timbre-based speaker clustering is an estimate, not robust diarization.
- Transcription and timestamps can be inaccurate, especially for young voices, noisy recordings, accents, and overlapping speech.
- Wait-time measurement is limited by transcription segment boundaries.
- Screen analysis measures visual changes rather than the pedagogical value of the content.
- The development server is loopback-bound by default; production use requires appropriate authentication, TLS, access control, retention rules, and security/privacy review.
