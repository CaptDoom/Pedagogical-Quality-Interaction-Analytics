# ClassLens project description

## Summary

**ClassLens — Classroom Interaction & Pedagogy Analytics** is a local-first application for reviewing recorded online lessons. It analyzes audio and optional video to produce an evidence-oriented report containing an overall rubric label, dimension scores, estimated teacher/student participation, a transcript, and timestamped observations.

## Problem and intended users

Reviewing a recorded lesson manually can be time-consuming and subjective. ClassLens organizes measurable signals and transcript evidence in one dashboard so educators, instructional coaches, researchers, and school teams can identify moments that may merit closer review. It is designed as a decision-support and reflection aid—not as an automated authority or substitute for professional judgment.

## How it works

1. A user uploads a media file or provides an eligible link.
2. FFmpeg validates media and extracts mono audio. Optional OpenCV sampling measures visual screen changes.
3. faster-whisper transcribes speech on the host machine.
4. Acoustic features and clustering estimate speaker turns; the longer-speaking voice is treated as the teacher when clustering is sufficiently confident.
5. The application measures signals such as talk share, questions, response patterns, speaking pace, praise, dead air, audio dropouts, and screen changes.
6. A configurable rubric converts signals into dimension scores, an aggregate score, and a label. Observations reference measured values and transcript timestamps.
7. If enabled, an LLM provider may review the transcript and aggregate metrics for qualitative indicators such as clarity, rapport, and content accuracy.

## Technical stack

- Python, FastAPI, and Uvicorn for the application and asynchronous job API.
- A single-page HTML/JavaScript dashboard.
- faster-whisper, FFmpeg/ffprobe, librosa, scikit-learn, NumPy, and OpenCV for media processing and signal extraction.
- Optional Gemini, OpenAI-compatible, Anthropic, or local Ollama review.

## Privacy model

Audio transcription and measured feature extraction are local by default. Uploaded copies and intermediate files are temporary and removed after a job. Enabling an external LLM sends transcript text and aggregate metrics to that selected service; the recording and sampled video frames are not sent by this application. Do not process recordings without appropriate authorization and consent. Recordings may contain children’s personal data.

## Scope and limitations

ClassLens produces approximate signals from noisy, incomplete inputs. Timbre-based speaker separation is not robust diarization, transcription may be inaccurate, wait-time measurement has known limitations, visual change is not a measure of instructional value, and rubric calibration is currently based on a very small labeled set. Scores should be treated as prompts for human review rather than definitive judgments or employment/disciplinary evidence.

## Repository and documentation

- [README.md](README.md): installation, configuration, API overview, tests, and project caveats.
- [USER_GUIDE.md](USER_GUIDE.md): step-by-step dashboard usage, report interpretation, privacy, and troubleshooting.
