# ClassLens user guide

This guide covers installing and running ClassLens, submitting a lesson recording, understanding a report, and troubleshooting common issues. Start with [README.md](README.md) for quick installation and configuration.

## 1. Before you begin

- Use a supported Windows, macOS, or Linux environment with Python 3.10+.
- Install FFmpeg and ensure both `ffmpeg` and `ffprobe` are on `PATH`.
- Install project dependencies with `pip install -r requirements.txt` in your active environment.
- Classroom recordings may contain sensitive information or personal data about minors. Use only recordings you are authorized to process. Keep the application local unless you have reviewed access control and deployment security.

Start the application from the project directory with `python server.py`, then open `http://127.0.0.1:8000`.

## 2. Submit a recording

1. Choose **Assess Classroom Recording** or **New Assessment**.
2. Select one of the input tabs:
   - **File Upload:** choose an audio or video file on your computer.
   - **URL / Link:** paste a direct media URL, a Google Drive sharing link, or a URL supported by yt-dlp. The host must be able to access the media.
   - **Demo Samples:** select a supported media file already available to the server.
3. Confirm the selected recording in the staging panel.
4. The hosted dashboard uses Whisper `tiny` and deterministic, audio-only analysis to limit CPU and memory use. Frame extraction, speaker separation, and external AI review are disabled.
5. Set the maximum analysis duration (1–15 minutes). Longer media is truncated to this duration.
6. If the server administrator configured `APP_PASSWORD`, enter that password in the application password field.
7. Click **Start Assessment** and keep the page open while status messages update. The report opens automatically when the job completes. Temporary HTTP 5xx/network errors are retried automatically; if the service restarts, submit the recording again.

## 3. Read the report

- **Overall Pedagogical Index:** a weighted score from 1 to 5 and a rubric label. Treat the label as a screening signal, not a verdict.
- **Confidence:** a qualitative indication from the scoring pipeline; it does not guarantee correctness.
- **Student Talk Share / Questions / Monologue / Feedback:** summaries of measured signals extracted from the transcript and timing.
- **Strengths and Concerns:** observations with evidence. Check the transcript and source recording around the cited time.
- **Dimension Breakdown:** weighted rubric dimensions and their data source.
- **Synchronized Transcript:** select a timestamp to seek the available video; use search and speaker filters to locate turns.
- **Signal Inspector:** expand the table to inspect measured feature values.
- **Copy Summary / Export PDF:** copy a text summary or use the browser print dialog to save a report.

When speaker identification is uncertain, speaker labels and student participation signals may be misleading. The report should flag this, and you should rely on the source recording rather than interpreting uncertain estimates as facts.

## 4. Privacy and data handling

- Media processing and transcription happen on the machine running ClassLens.
- A per-run temporary workspace holds the uploaded copy and extracted audio; application code removes it after the run.
- The hosted dashboard uses deterministic scoring only and does not send transcripts, audio, or video to an external AI provider.
- Your API key and `APP_PASSWORD` are credentials. Keep them in the ignored `.env` file or a password manager; do not put secrets into source control, screenshots, or bug reports.
- Do not upload recordings to public hosting or use links that grant broader access than intended.
- Do not expose the development server directly to the public internet. Production deployment requires authentication, TLS, access controls, retention policies, and an institutional privacy/security review.

## 5. Troubleshooting

| Symptom | Checks and next steps |
| --- | --- |
| App reports that FFmpeg is missing | Install FFmpeg, reopen the terminal so `PATH` refreshes, and verify `ffmpeg -version` and `ffprobe -version`. |
| Upload or link says the file is not media / has no audio | Confirm the file plays locally and contains an audio track. Try a supported audio/video container or another source URL. |
| Link download fails | Confirm the link is reachable from the app host and allows download without an interactive login. For Drive, enable link access for the intended audience. |
| Analysis takes a long time or runs out of memory | Use a shorter duration. The dashboard already uses the smallest Whisper model and disables frame extraction, speaker separation, and external AI review. The first model run may need to download model files. |
| Speech transcript is inaccurate | Improve audio quality and manually verify important evidence. Young voices, accents, overlapping speech, and background noise are challenging; the hosted dashboard intentionally has no slower model selector. |
| Assessment status briefly returns HTTP 502/503/504 | Keep the page open; it retries temporary server/gateway errors with increasing delays. If the service restarted, the in-memory job is lost and must be resubmitted. |
| Assessment returns HTTP 401 | The server requires `APP_PASSWORD`; enter the configured password in the application password field. |
| Sample list is empty | Sample media is intentionally excluded from source control. Add authorized local media to a supported sample directory on the server. |
| Job ID is unknown after a restart | Jobs are stored in memory and do not persist across service restarts. Submit the recording again. |

## 6. API users

Interactive API documentation is available at `http://127.0.0.1:8000/api`. `POST /assess` accepts multipart form data and returns a job ID. Poll `GET /jobs/{id}` until the status is `done` or `error`. `GET /api/latest` returns the latest report in the current server process. Keep the app and API private; uploaded content and transcript reports are sensitive.

## 7. Known limitations

- Rubric calibration uses a small, imbalanced labeled sample set and has no labeled Poor recording; the Poor cutoff is untested and performance on other classes is not established.
- The hosted dashboard does not separate speakers; participation estimates are unavailable.
- Transcript text and timestamps can be wrong. Review the recording before relying on quoted evidence.
- Word-level transcription confidence is exposed in the transcript and low-confidence text contributes less to lexical metrics. Audio activity, rather than ASR segment edges, is used for silence and wait-time estimates; neither is a human-verified annotation.
- Signal lexicons are mainly English, so language-dependent measures can be less reliable for other languages.
- The hosted dashboard does not analyze video frames or screen text.
