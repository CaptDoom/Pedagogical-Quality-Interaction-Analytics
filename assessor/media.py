from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

# Fall back to bundled ffmpeg if present in .tools/ffmpeg/bin or bin/
_base_dir = Path(__file__).resolve().parent.parent
for _cand_bin in [_base_dir / ".tools" / "ffmpeg" / "bin", _base_dir / "bin"]:
    if _cand_bin.is_dir() and str(_cand_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_cand_bin) + os.pathsep + os.environ.get("PATH", "")

MEDIA_EXT = {".mp4", ".mkv", ".mov", ".webm", ".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flv", ".wmv", ".avi", ".ts"}
# ffprobe format_name -> sensible file extension (for links that ship without one)
FORMAT_EXT = {
    "matroska,webm": ".mkv",
    "mov,mp4,m4a,3gp,3g2,mj2": ".mp4",
    "mp3": ".mp3",
    "wav": ".wav",
    "ogg": ".ogg",
    "aac": ".aac",
    "flac": ".flac",
}
CT_EXT = {
    "video/mp4": ".mp4", "video/webm": ".webm", "video/x-matroska": ".mkv", "video/quicktime": ".mov",
    "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mp4": ".m4a",
    "audio/aac": ".aac", "audio/ogg": ".ogg",
}


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def probe(path: str) -> dict:
    """ffprobe the file: {duration, has_audio, has_video, format}. Raises a readable error if unreadable."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}  # degradation: skip validation rather than block (ffmpeg availability is checked later)
    r = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"'{os.path.basename(path)}' is not a media file ffmpeg can read: {r.stderr.strip()[:300]}")
    try:
        info = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"'{os.path.basename(path)}' could not be inspected by ffprobe.")
    streams = info.get("streams", [])
    return {
        "duration": float(info.get("format", {}).get("duration", 0) or 0),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "has_video": any(s.get("codec_type") == "video" for s in streams),
        "format": info.get("format", {}).get("format_name", ""),
    }


def ingest(source: str, outdir: str) -> tuple[str, dict]:
    """Accept a local path or any link and return (local_file, probe_info).

    Links dispatch as: Google Drive share -> direct media URL -> yt-dlp (YouTube & friends) -> plain HTTP.
    The result is always ffprobe-checked, so a bad source fails here with a clear message,
    not deep inside the pipeline.
    """
    src = os.path.expanduser(source.strip().strip('"'))
    if os.path.isfile(src):
        path = os.path.abspath(src)
    elif urllib.parse.urlparse(src).scheme in ("http", "https", "ftp"):
        path = download(src, outdir)
    else:
        raise RuntimeError(f"No such recording: {source!r}")
    info = probe(path)
    if info and not info["has_audio"]:
        raise RuntimeError(f"'{os.path.basename(path)}' has no audio track, so there is no class to assess.")
    return path, info


def _stream_http(url: str, outdir: str) -> str:
    """Streamed download with redirects, filename hints and two retries (class videos are large)."""
    last: Exception | None = None
    for attempt in range(3):
        dest = os.path.join(outdir, "recording")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ClassLens)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                headers = dict(r.headers)
                ctype = (headers.get("Content-Type") or "").split(";")[0].strip()
                name = ""
                disp = headers.get("Content-Disposition", "")
                if "filename=" in disp:
                    name = disp.split("filename=")[-1].strip().strip('"').strip(";")
                if not name:
                    name = os.path.basename(urllib.parse.urlparse(url).path)
                if name:
                    dest = os.path.join(outdir, os.path.basename(name))
                with open(dest, "wb") as fh:
                    shutil.copyfileobj(r, fh, 1 << 20)
            size = os.path.getsize(dest)
            if size < 1024:
                raise RuntimeError(f"server returned only {size} bytes")
            if not os.path.splitext(dest)[1]:  # link shipped without an extension
                dest2 = dest + CT_EXT.get(ctype, ".bin")
                os.replace(dest, dest2)
                dest = dest2
            return dest
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"could not download {url}: {last}")


def download(url: str, outdir: str) -> str:
    """Fetch a recording from a link: Drive share, direct media URL, yt-dlp platform, or plain HTTP."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    url = url.strip()

    if "drive.google.com" in url or "docs.google.com" in url:
        try:
            import gdown

            m = re.search(r"[-_\w]{25,}", url)
            file_id = m.group(0) if m else None
            dest = os.path.join(outdir, "recording.mp4")
            out = None
            if file_id:
                try:
                    out = gdown.download(id=file_id, output=dest, quiet=True)
                except Exception:
                    out = None
            if not out:
                out = gdown.download(url=url, output=dest, quiet=True)
            if out and os.path.exists(out) and os.path.getsize(out) > 0:
                return out
            raise RuntimeError("gdown returned nothing")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Google Drive download failed ({e}). Check the link is shared as 'anyone with the link'.") from e

    if _looks_direct(url):
        return _stream_http(url, outdir)

    # platform URLs (YouTube, Vimeo, ...): yt-dlp, capped at 720p (audio + frames are all we need)
    try:
        import yt_dlp

        opts = {
            "outtmpl": os.path.join(outdir, "%(title).80s.%(ext)s"),
            "quiet": True, "noprogress": True, "retries": 3,
            "format": "bv*[height<=720]+ba/b[height<=720]/b",
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
            if not os.path.exists(path) and info.get("ext"):
                path = os.path.splitext(path)[0] + "." + info["ext"]
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path
    except Exception:  # noqa: BLE001
        pass

    # last resort: treat it as a plain file URL
    return _stream_http(url, outdir)


def _looks_direct(url: str) -> bool:
    path = urllib.parse.urlparse(url).path.lower()
    return any(path.endswith(e) for e in MEDIA_EXT)


def extract_audio(src: str, dst: str, max_minutes: float | None = None,
                  denoised_dst: str | None = None, denoise: bool = False) -> None:
    cmd = ["ffmpeg", "-y", "-i", src, "-map", "0:a:0?", "-vn", "-ac", "1", "-ar", "16000"]
    if denoise and not denoised_dst:
        cmd += ["-af", "highpass=f=70,afftdn=nf=-25"]
    if max_minutes:
        cmd += ["-t", str(int(max_minutes * 60))]
    cmd.append(dst)
    if denoised_dst:
        cmd += ["-map", "0:a:0?", "-vn", "-af", "highpass=f=70,afftdn=nf=-25", "-ac", "1", "-ar", "16000"]
        if max_minutes:
            cmd += ["-t", str(int(max_minutes * 60))]
        cmd.append(denoised_dst)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst) or (denoised_dst and not os.path.exists(denoised_dst)):
        raise RuntimeError("ffmpeg could not read audio from this file:\n" + r.stderr[-600:])


def sample_frames(src: str, step_s: float = 10.0, max_frames: int = 150, max_seconds: float | None = None):
    """Return ([(t, small grayscale frame)], step) or ([], step) if the file has no video."""
    try:
        import cv2
    except ImportError:
        return [], step_s
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return [], step_s
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if fps <= 0 or n <= 0:
        cap.release()
        return [], step_s
    dur = n / fps
    if max_seconds:
        dur = min(dur, max_seconds)
    step = max(step_s, dur / max_frames)
    out = []
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY)
        out.append((t, g.astype(np.float32)))
        t += step
    cap.release()
    return out, step
