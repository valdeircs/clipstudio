"""Local video rendering with optional Apify transcript and AI analysis adapters.

YouTube access uses yt-dlp; speech recognition uses a locally cached Whisper
model. Local ranking uses a transparent heuristic; the API workflow uses one
selected AI provider. Suggestions are not predictions of virality. Captions are rendered with Pillow so an FFmpeg libass build is not
required.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Callable
from urllib.parse import parse_qs, urlparse
import uuid

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps


Progress = Callable[[str, int, str], None]
MAX_DURATION = 7200
MAX_SOURCE_BYTES = 2 * 1024**3
LANGUAGES = {"auto", "en", "pt", "es", "fr", "de", "it", "nl", "ja", "ko", "zh", "ar", "hi", "ru", "uk", "tr", "pl"}


class VideoError(RuntimeError):
    """A concise error safe to show in the local application."""


def _binary(name: str) -> str | None:
    local = Path(__file__).resolve().parent / ".venv" / "bin" / name
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    found = shutil.which(name)
    # Keep custom PATH overrides; prefer the complete Homebrew build over its
    # standard link, which can break when shared codec libraries are upgraded.
    if name in {"ffmpeg", "ffprobe"} and found in {
        None, f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}",
    }:
        for directory in ("/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"):
            candidate = Path(directory) / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return found or (str(Path("/opt/homebrew/bin") / name) if (Path("/opt/homebrew/bin") / name).is_file() else None)


def _models() -> list[str]:
    directory = Path.home() / ".cache" / "whisper"
    return [name for name in ("base", "tiny", "small", "base.en", "medium", "large-v3-turbo") if (directory / f"{name}.pt").is_file()]


def capabilities() -> dict:
    installed = {name: bool(_binary(name)) for name in ("ffmpeg", "ffprobe", "yt-dlp", "whisper")}
    models = _models()
    return {
        **installed,
        "ffmpeg": installed["ffmpeg"],
        "ffprobe": installed["ffprobe"],
        "yt_dlp": installed["yt-dlp"],
        "whisper": installed["whisper"],
        "models": models,
        "cached_models": models,
        "local_transcription": bool(installed["whisper"] and models),
        "youtube": bool(installed["yt-dlp"] and installed["ffmpeg"]),
        "render": bool(installed["ffmpeg"] and installed["ffprobe"]),
        "free": True,
        "max_duration_seconds": MAX_DURATION,
        "max_source_bytes": MAX_SOURCE_BYTES,
        "analysis_method": "Local transcript scoring: complete thoughts, questions, useful explanations and speech density.",
    }


def _number(value, default: float, minimum: float, maximum: float, name: str) -> float:
    try:
        number = float(default if value is None else value)
    except (TypeError, ValueError):
        raise VideoError(f"{name} must be a number.") from None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise VideoError(f"{name} must be between {minimum:g} and {maximum:g}.")
    return number


def _run(args: list[str], cwd: Path, timeout: int = 300, failure: str = "Video processing failed.") -> str:
    env = dict(os.environ)
    ffmpeg = _binary("ffmpeg")
    # yt-dlp and Whisper may launch FFmpeg themselves; use the same binary as
    # direct rendering and leave the caller's other custom commands in order.
    media_path = str(Path(ffmpeg).parent) + os.pathsep if ffmpeg else ""
    env["PATH"] = media_path + env.get("PATH", "") + ":/opt/homebrew/bin:/usr/local/bin"
    env["OMP_NUM_THREADS"] = "2"
    try:
        result = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        raise VideoError(f"{failure} The operation timed out; try a shorter source video.") from None
    except OSError:
        raise VideoError(f"{failure} A required command could not be started.") from None
    if result.returncode:
        # Keep details in a local diagnostic log, never in browser error HTML.
        with (cwd / "processing.log").open("a", encoding="utf-8") as log:
            log.write(f"\n[{Path(args[0]).name}]\n{result.stderr[-12000:]}\n")
        details = result.stderr.lower()
        if "http error 403" in details and "yt-dlp" in Path(args[0]).name:
            raise VideoError("YouTube rejected the video download. Update ClipStudio's yt-dlp as described in README.md, or upload the original video file.")
        if any(word in details for word in ("sign in", "confirm you're", "login required", "private video", "members-only", "video unavailable")):
            raise VideoError("YouTube restricted this video. Upload a video file you can access instead.")
        if "requested format is not available" in details:
            raise VideoError("YouTube did not provide a downloadable format. Upload the source file instead.")
        if any(word in details for word in ("could not resolve", "nodename nor servname", "temporary failure in name resolution", "network is unreachable")):
            raise VideoError("The internet connection could not reach YouTube. Check your connection or upload a local video.")
        raise VideoError(failure)
    return result.stdout


def _require(name: str) -> str:
    binary = _binary(name)
    if not binary:
        raise VideoError(f"{name} is not installed. See the setup instructions in README.md.")
    return binary


def _probe(path: Path) -> dict:
    raw = _run([_require("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], path.parent, failure="The source file could not be read as a video.")
    try:
        data = json.loads(raw)
        streams = data.get("streams", [])
        video = next(s for s in streams if s.get("codec_type") == "video")
        duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    except (ValueError, TypeError, StopIteration):
        raise VideoError("The source must be a video file with a readable duration.") from None
    if not math.isfinite(duration) or duration <= 0 or duration > MAX_DURATION:
        raise VideoError("Use a video longer than zero seconds and no longer than two hours.")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {"duration": duration, "width": video.get("width", 0), "height": video.get("height", 0), "has_audio": audio is not None, "video_codec": video.get("codec_name"), "pixel_format": video.get("pix_fmt"), "audio_codec": audio.get("codec_name") if audio else None, "format_name": data.get("format", {}).get("format_name", "")}


def _normalize_source(folder: Path, source: Path, media: dict, progress: Progress) -> tuple[Path, dict]:
    """Make browser playback reliable without re-encoding compatible video."""
    video_compatible = media["video_codec"] == "h264" and media["pixel_format"] in {"yuv420p", "yuvj420p"}
    audio_compatible = not media["has_audio"] or media["audio_codec"] == "aac"
    mp4_container = source.suffix.lower() == ".mp4" and "mp4" in media["format_name"].split(",")
    if video_compatible and audio_compatible and mp4_container:
        return source, media
    action = "Preparing" if video_compatible and audio_compatible else "Converting"
    progress("normalize", 18, f"{action} a browser-compatible MP4 preview on your computer…")
    partial = folder / f"source-preview-{uuid.uuid4().hex[:10]}.partial.mp4"
    target = folder / "source-preview.mp4"
    args = [_require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-threads", "2", "-i", source.name, "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn"]
    if video_compatible:
        args += ["-c:v", "copy"]
    else:
        args += ["-vf", "scale=w='min(iw,1920)':h='min(ih,1920)':force_original_aspect_ratio=decrease:force_divisible_by=2,fps=30", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    args += ["-c:a", "copy"] if audio_compatible else ["-c:a", "aac", "-b:a", "160k"]
    args += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", partial.name]
    timeout = max(300, min(10800, int(media["duration"] * 1.5 + 120)))
    process = None
    try:
        process = subprocess.Popen(args, cwd=str(folder), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        started = time.monotonic()
        previous_percent = -1
        while True:
            try:
                output, errors = process.communicate(timeout=2)
                break
            except subprocess.TimeoutExpired as pending:
                if time.monotonic() - started > timeout:
                    process.kill()
                    process.communicate()
                    raise VideoError("Preparing the browser preview took too long. Try a shorter video or an H.264 MP4 source.") from None
                output = pending.output or ""
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
                timestamps = re.findall(r"out_time_us=(\d+)", output)
                if timestamps:
                    fraction = min(1.0, int(timestamps[-1]) / 1_000_000 / media["duration"])
                    percent = round(fraction * 100)
                    if percent != previous_percent:
                        progress("normalize", 18 + round(fraction * 15), f"{action} a browser-compatible MP4 preview… {percent}%")
                        previous_percent = percent
        if process.returncode:
            with (folder / "processing.log").open("a", encoding="utf-8") as log:
                log.write(f"\n[normalize]\n{errors[-12000:]}\n")
            raise VideoError("This video could not be converted for browser playback. Try uploading an H.264 MP4 file.")
        normalized = _probe(partial)
        if normalized["video_codec"] != "h264" or (normalized["has_audio"] and normalized["audio_codec"] != "aac"):
            raise VideoError("The source could not be converted to a browser-compatible video.")
        os.replace(partial, target)
        progress("normalize", 33, "Browser-compatible preview is ready.")
        return target, normalized
    except OSError:
        raise VideoError("The browser preview could not be created. Check the available disk space and FFmpeg installation.") from None
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        partial.unlink(missing_ok=True)


def _youtube_url(value: str) -> str:
    try:
        parsed = urlparse(str(value).strip())
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise ValueError
        host = (parsed.hostname or "").lower()
        parts = [part for part in parsed.path.split("/") if part]
        if host == "youtu.be" and len(parts) == 1:
            video_id = parts[0]
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
            if parsed.path.rstrip("/") == "/watch":
                ids = parse_qs(parsed.query).get("v", [])
                if len(ids) != 1:
                    raise ValueError
                video_id = ids[0]
            elif len(parts) == 2 and parts[0] in {"shorts", "embed", "live"}:
                video_id = parts[1]
            else:
                raise ValueError
        else:
            raise ValueError
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError
    except (ValueError, TypeError):
        raise VideoError("Paste an HTTPS YouTube video link, such as https://www.youtube.com/watch?v=VIDEO_ID.") from None
    return "https://www.youtube.com/watch?v=" + video_id


def _download(folder: Path, url: str, language: str, progress: Progress) -> tuple[Path, dict]:
    url = _youtube_url(url)
    ytdlp = _require("yt-dlp")
    common = [ytdlp, "--ignore-config", "--no-playlist", "--no-progress", "--socket-timeout", "20", "--retries", "2"]
    runtime = _binary("deno")
    if runtime:
        common += ["--js-runtimes", f"deno:{runtime}"]
    elif _binary("node"):
        common += ["--js-runtimes", f"node:{_binary('node')}"]
    progress("download", 5, "Checking the YouTube video and available captions…")
    raw = _run(common + ["--dump-single-json", "--skip-download", url], folder, timeout=180, failure="YouTube could not be opened. Try uploading the video file instead.")
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        raise VideoError("YouTube returned unreadable video information.") from None
    if info.get("is_live") or info.get("live_status") == "is_live":
        raise VideoError("Live broadcasts must finish before they can be clipped.")
    duration = info.get("duration")
    if duration is None or not 0 < float(duration) <= MAX_DURATION:
        raise VideoError("Choose a YouTube video no longer than two hours.")
    captions = {**info.get("automatic_captions", {}), **info.get("subtitles", {})}
    original = str(info.get("language") or "").split("-")[0]
    if language != "auto":
        preferred = [language, f"{language}-orig"]
    else:
        preferred = ([original, f"{original}-orig"] if original else []) + [k for k in captions if k.endswith("-orig")] + ["en", "pt", "es", "fr", "de"]
    selected = next((key for key in preferred if key in captions), None)
    args = common + ["--concurrent-fragments", "4", "--max-filesize", str(MAX_SOURCE_BYTES), "--match-filter", f"duration <= {MAX_DURATION}", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b", "--merge-output-format", "mp4", "--remux-video", "mp4", "--ffmpeg-location", str(Path(_require("ffmpeg")).parent), "-o", "source.%(ext)s"]
    progress("download", 12, f"Downloading {str(info.get('title') or 'your video')[:150]} ({int(duration) // 60}:{int(duration) % 60:02d})…")
    _run(args + [url], folder, timeout=1800, failure="The video could not be downloaded. Try uploading a source file instead.")
    source = folder / "source.mp4"
    if not source.is_file():
        raise VideoError("YouTube did not produce a usable video file. Upload a source file instead.")
    if source.stat().st_size > MAX_SOURCE_BYTES:
        raise VideoError("The downloaded video is larger than 2 GB. Choose a shorter source.")
    if selected:
        progress("captions", 16, "Checking for existing captions before local transcription…")
        try:
            # Subtitles have independent rate limits. Their failure must never
            # discard a usable source video or prevent local transcription.
            _run(common + ["--skip-download", "--write-subs", "--write-auto-subs", "--sub-langs", selected, "--sub-format", "json3", "-o", "source.%(ext)s", url], folder, timeout=120, failure="YouTube captions were unavailable; using local transcription.")
        except VideoError:
            progress("captions", 17, "YouTube captions are unavailable. The video will be transcribed locally.")
    info["selected_caption_language"] = selected.split("-")[0] if selected else None
    return source, info


def _clean_segments(segments: list, duration: float) -> list[dict]:
    cleaned = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start + end):
            continue
        start, end = max(0.0, start), min(duration, end)
        text = re.sub(r"\s+", " ", str(segment.get("text", ""))).strip()
        if text and end > start:
            item = {"start": round(start, 3), "end": round(end, 3), "text": text}
            words = []
            for word in segment.get("words", []):
                try:
                    ws, we = float(word["start"]), float(word["end"])
                    if math.isfinite(ws + we) and we > ws and 0 <= ws < duration:
                        words.append({"start": round(ws, 3), "end": round(min(duration, we), 3), "word": str(word.get("word", ""))})
                except (KeyError, TypeError, ValueError):
                    pass
            if words:
                item["words"] = words
            cleaned.append(item)
    cleaned.sort(key=lambda s: (s["start"], s["end"]))
    # Rolling automatic subtitles can overlap; end the preceding cue at the new cue.
    for previous, current in zip(cleaned, cleaned[1:]):
        if previous["end"] > current["start"] and current["start"] > previous["start"]:
            previous["end"] = current["start"]
    return cleaned


def _json3(path: Path, duration: float) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    result = []
    for event in data.get("events", []):
        parts = event.get("segs", [])
        text = "".join(str(part.get("utf8", "")) for part in parts).strip()
        if not text:
            continue
        start = float(event.get("tStartMs", 0)) / 1000
        end = start + float(event.get("dDurationMs", 0)) / 1000
        words = []
        for index, part in enumerate(parts):
            word = str(part.get("utf8", "")).strip()
            ws = start + float(part.get("tOffsetMs", 0)) / 1000
            we = start + float(parts[index + 1].get("tOffsetMs", (end - start) * 1000)) / 1000 if index + 1 < len(parts) else end
            if word and we > ws:
                words.append({"start": ws, "end": we, "word": word})
        result.append({"start": start, "end": end, "text": text, "words": words})
    return _clean_segments(result, duration)


def _transcribe(folder: Path, source: Path, language: str, progress: Progress) -> tuple[list, str]:
    available = _models()
    if not available or not _binary("whisper"):
        raise VideoError("No local Whisper model is available. Follow README.md to install Whisper and download a model once.")
    model = "base" if "base" in available else available[0]
    if model.endswith(".en") and language not in {"auto", "en"}:
        raise VideoError("This installation has only an English Whisper model. Download the multilingual base model for this language.")
    progress("transcribe", 35, f"Transcribing locally with Whisper {model}; longer videos may take several minutes…")
    _run([_require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", source.name, "-vn", "-ar", "16000", "-ac", "1", "audio.wav"], folder, timeout=600, failure="The audio track could not be extracted.")
    args = [_require("whisper"), "audio.wav", "--model", model, "--model_dir", str(Path.home() / ".cache" / "whisper"), "--device", "cpu", "--fp16", "False", "--threads", "2", "--output_dir", ".", "--output_format", "json", "--word_timestamps", "True", "--verbose", "False"]
    if language != "auto":
        args += ["--language", language]
    try:
        _run(args, folder, timeout=10800, failure="Local speech transcription failed. Check processing.log in the project folder.")
        data = json.loads((folder / "audio.json").read_text(encoding="utf-8"))
    finally:
        (folder / "audio.wav").unlink(missing_ok=True)
    return data.get("segments", []), str(data.get("language", language))


def _srt_time(seconds: float) -> str:
    millis = round(max(0, seconds) * 1000)
    hours, remainder = divmod(millis, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def _write_srt(path: Path, segments: list[dict]) -> None:
    path.write_text("\n".join(f"{i}\n{_srt_time(s['start'])} --> {_srt_time(s['end'])}\n{s['text']}\n" for i, s in enumerate(segments, 1)), encoding="utf-8")


def _rank_clips(segments: list[dict], duration: float, length: float, count: int) -> list[dict]:
    candidates = []
    markers = ("because", "here's", "here is", "how to", "why", "mistake", "secret", "lesson", "learn", "three", "first", "instead", "imagine", "important", "porque", "como", "por que", "erro", "segredo", "aprendi", "primeiro", "importante", "dica", "entenda", "consejo", "aprende")
    sentence_end = re.compile(r"[.!?…][\"'”’)]*$")
    opening_cue = re.compile(r"^(?:why|how|what|when|where|who|imagine|notice|remember|here['’]s|here is|let['’]s|por que|por quê|como|veja|imagine|lembre|vamos|o que|quando|por qué|cómo|mira|recuerda)\b", re.IGNORECASE)

    def speech_text(text: str) -> str:
        return re.sub(r"\[[^\]]*\]", "", text).strip().lstrip(">").strip()

    for index, first in enumerate(segments):
        start = max(0, first["start"] - 0.12)
        previous = segments[index - 1] if index else None
        # Subtitle cues wrap wherever the screen fills, often in the middle of
        # a sentence. Prefer genuine thought boundaries while retaining every
        # cue as a fallback when captions contain no punctuation or pauses.
        boundary = previous is None or bool(sentence_end.search(speech_text(previous["text"]))) or first["start"] - previous["end"] >= .8
        starts_thought = bool(opening_cue.match(speech_text(first["text"])))
        start_quality = (14 if boundary else 4 if starts_thought else -20) + (8 if starts_thought else 0)
        if duration - start < min(10, length * .5):
            continue
        selected = []
        target_selection = None
        complete_end = False
        for segment in segments[index:]:
            if segment["start"] >= start + length * 1.2:
                break
            if selected and segment["end"] - start > min(90, length * 1.2):
                break
            selected.append(segment)
            elapsed = segment["end"] - start
            if elapsed >= length and target_selection is None:
                target_selection = list(selected)
            if elapsed >= length * .82 and sentence_end.search(speech_text(segment["text"])):
                complete_end = True
                break
        # Use the allowed 20% duration tolerance to finish a sentence. If no
        # ending exists in that window, keep the original target-length cut.
        if not complete_end and target_selection is not None:
            selected = target_selection
        if not selected:
            continue
        end = min(duration, selected[-1]["end"] + .15, start + 90)
        text = " ".join(item["text"] for item in selected)
        words = text.split()
        if len(words) < 5:
            continue
        lower = text.lower()
        signals = sum(bool(re.search(r"\b" + re.escape(marker) + r"\b", lower)) for marker in markers)
        complete = bool(sentence_end.search(speech_text(text)))
        question = "?" in text
        spoken = sum(min(s["end"], end) - max(s["start"], start) for s in selected)
        density = min(1, spoken / max(1, end - start))
        opening = not bool(re.match(r"^(and|but|so|then|also|e|mas|então|também)\b", lower))
        score = 35 + min(22, signals * 5) + 12 * complete + 6 * question + 12 * density + 6 * opening + start_quality - min(18, abs((end - start) - length) / length * 18)
        reasons = []
        if boundary:
            reasons.append("Starts after a sentence boundary or speech pause")
        elif starts_thought:
            reasons.append("Starts with a question or introductory phrase")
        if signals:
            reasons.append("Useful explanation or takeaway language")
        if question:
            reasons.append("Includes a question that can open a discussion")
        if complete:
            reasons.append("Ends at a sentence boundary")
        if density > .8:
            reasons.append("Consistent speech with little empty time")
        if not reasons:
            reasons.append("A compact passage near your requested duration")
        title = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
        if len(title) > 72:
            title = title[:69].rsplit(" ", 1)[0] + "…"
        candidates.append({"start": round(start, 2), "end": round(end, 2), "title": title, "text": text, "reason": "; ".join(reasons) + ". Review the context before posting.", "score": round(max(1, min(99, score))), "status": "suggested"})
    candidates.sort(key=lambda clip: (-clip["score"], clip["start"]))
    chosen = []
    for candidate in candidates:
        overlaps = [max(0, min(candidate["end"], other["end"]) - max(candidate["start"], other["start"])) / min(candidate["end"] - candidate["start"], other["end"] - other["start"]) for other in chosen]
        if any(overlap > .15 for overlap in overlaps):
            continue
        candidate["id"] = f"clip-{len(chosen) + 1}"
        chosen.append(candidate)
        if len(chosen) == count:
            break
    return chosen


def analyze_project(folder: Path, request: dict, progress: Progress) -> dict:
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if request.get("pipeline") in {"local_ai", "apify_ai"}:
        return analyze_api_project(folder, request, progress)
    language = str(request.get("language", "auto"))
    if language not in LANGUAGES:
        raise VideoError("Choose a supported transcription language.")
    length = _number(request.get("length", request.get("clip_length", 45)), 45, 15, 90, "Clip length")
    count = int(_number(request.get("count", request.get("clip_count", 3)), 3, 1, 8, "Number of clips"))
    info = {}
    if request.get("source_file"):
        original = Path(str(request["source_file"])).resolve()
        if not original.is_file() or original.stat().st_size > MAX_SOURCE_BYTES:
            raise VideoError("The uploaded video is missing or larger than 2 GB.")
        suffix = original.suffix.lower()
        suffix = suffix if suffix in {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"} else ".mp4"
        source = folder / ("source" + suffix)
        progress("import", 10, "Importing your video…")
        if original != source:
            shutil.copyfile(original, source)
        info["title"] = str(request.get("title") or original.stem)
        if request.get("metadata_file"):
            try:
                info.update(json.loads(Path(request["metadata_file"]).read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
    else:
        source, info = _download(folder, str(request.get("url", request.get("youtube_url", ""))), language, progress)
    media = _probe(source)
    source, media = _normalize_source(folder, source, media, progress)
    duration = media["duration"]
    segments = []
    detected_language = language
    transcript_origin = ""
    if request.get("transcript_file"):
        try:
            provided = json.loads(Path(request["transcript_file"]).read_text(encoding="utf-8"))
            segments = _clean_segments(provided.get("segments", []) if isinstance(provided, dict) else provided, duration)
            detected_language = provided.get("language", language) if isinstance(provided, dict) else language
            transcript_origin = "Provided transcript"
        except (OSError, ValueError, AttributeError):
            raise VideoError("The supplied transcript could not be read.") from None
    if not segments:
        for subtitle in sorted(folder.glob("source.*.json3")):
            segments = _json3(subtitle, duration)
            if segments:
                transcript_origin = "YouTube captions"
                detected_language = info.get("selected_caption_language") or language
                break
    if not segments:
        if not media["has_audio"]:
            raise VideoError("This video has no audio track. A spoken video is needed for transcript-based highlights.")
        raw_segments, detected_language = _transcribe(folder, source, language, progress)
        segments = _clean_segments(raw_segments, duration)
        transcript_origin = "Local Whisper transcription"
    if not segments:
        raise VideoError("No speech was found. Try a video with clear spoken audio.")
    progress("analyze", 85, "Finding complete thoughts and useful explanations in the transcript…")
    clips = _rank_clips(segments, duration, length, count)
    if not clips:
        raise VideoError("There was not enough spoken content to suggest a clip. Try a longer spoken video.")
    _write_srt(folder / "transcript.srt", segments)
    (folder / "transcript.txt").write_text("\n".join(s["text"] for s in segments), encoding="utf-8")
    (folder / "transcript.json").write_text(json.dumps({"language": detected_language, "segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("ready", 100, f"Found {len(clips)} suggested clips. Review the wording and crop, then export.")
    return {"source_url": _youtube_url(request["url"]) if request.get("url") else None, "pipeline": "local", "title": str(info.get("title") or "Untitled video")[:200], "duration": round(duration, 3), "language": detected_language, "source": source.name, "segments": segments, "clips": clips, "width": media["width"], "height": media["height"], "transcript": "transcript.txt", "transcript_srt": "transcript.srt", "transcript_origin": transcript_origin, "analysis_method": "Heuristic transcript ranking — explanation keywords, sentence endings, questions and speech density. Scores are editorial suggestions, not virality predictions."}


def _font(size: int, bold: bool = True):
    choices = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"]
    for path in choices:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def _wrap(text: str, font, width: int, max_lines: int | None = None) -> list[str]:
    draw = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    lines, line = [], ""
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and draw.textlength(lines[-1] + "…", font=font) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "…"
    return lines or [""]


def _clip_cues(segments: list[dict], start: float, end: float) -> list[dict]:
    cues = []
    for segment in segments:
        if segment["end"] <= start or segment["start"] >= end:
            continue
        words = segment.get("words")
        if words:
            # Some subtitle providers call a whole phrase a "word". Split those
            # entries using estimated times so no spoken text is truncated.
            # JSON3 word durations can extend past the next word or the cue.
            # Normalize those bounds before splitting/selecting so a cut at the
            # next word never brings the preceding word into the new clip.
            expanded = []
            timed_words = sorted(words, key=lambda word: word["start"])
            for word_index, word in enumerate(timed_words):
                tokens = word["word"].split()
                word_start = max(segment["start"], word["start"])
                word_end = min(segment["end"], word["end"])
                if word_index + 1 < len(timed_words):
                    word_end = min(word_end, timed_words[word_index + 1]["start"])
                if word_end <= word_start:
                    continue
                span = word_end - word_start
                for index, token in enumerate(tokens):
                    expanded.append({"word": token, "start": word_start + span * index / len(tokens), "end": word_start + span * (index + 1) / len(tokens)})
            selected = [word for word in expanded if word["end"] > start and word["start"] < end]
            for offset in range(0, len(selected), 7):
                group = selected[offset:offset + 7]
                cues.append({"start": max(0, group[0]["start"] - start), "end": min(end - start, group[-1]["end"] - start), "text": " ".join(word["word"].strip() for word in group)})
        else:
            ss, ee = segment["start"], segment["end"]
            tokens = segment["text"].split()
            groups = [tokens[i:i + 7] for i in range(0, len(tokens), 7)]
            for index, group in enumerate(groups):
                cue_start = ss + (ee - ss) * index / len(groups)
                cue_end = ss + (ee - ss) * (index + 1) / len(groups)
                if cue_end > start and cue_start < end:
                    cues.append({"start": max(0, cue_start - start), "end": min(end - start, cue_end - start), "text": " ".join(group)})
    cues.sort(key=lambda cue: cue["start"])
    for prev, nxt in zip(cues, cues[1:]):
        prev["end"] = min(prev["end"], nxt["start"])
    return [cue for cue in cues if cue["end"] - cue["start"] >= .04 and cue["text"].strip()]


CAPTION_STYLES = {
    "bold": {"label": "Bold yellow", "bold": True, "scale": .060, "color": "#FFE266"},
    "clean": {"label": "Clean white", "bold": False, "scale": .051, "color": "#FFFFFF"},
    "boxed": {"label": "White cards", "bold": True, "scale": .056, "color": "#10131A"},
    "outline": {"label": "Strong outline", "bold": True, "scale": .062, "color": "#FFFFFF"},
    "neon": {"label": "Neon mint", "bold": True, "scale": .058, "color": "#A9FFE0"},
    "minimal": {"label": "Minimal", "bold": False, "scale": .048, "color": "#FFFFFF"},
}
CAPTION_SIZES = {"small": .82, "medium": 1.0, "large": 1.2}
CAPTION_POSITIONS = {"lower": .71, "middle": .47, "top": .19}


def _caption_wrap(text: str, font, max_width: int) -> list[str]:
    """Wrap every spoken word, including long words, without an ellipsis."""
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    parts = []
    for word in str(text).split():
        part = ""
        for char in word:
            if part and draw.textlength(part + char, font=font) > max_width:
                parts.append(part)
                part = char
            else:
                part += char
        if part:
            parts.append(part)
    lines, line = [], ""
    for part in parts:
        trial = f"{line} {part}".strip()
        if line and draw.textlength(trial, font=font) > max_width:
            lines.append(line)
            line = part
        else:
            line = trial
    if line:
        lines.append(line)
    return lines or [""]


def _caption_image(text: str, width: int, height: int, style: str, size: str = "medium", position: str = "lower") -> Image.Image:
    """Render one phrase inside the social-video safe area; timing lives elsewhere."""
    preset = CAPTION_STYLES[style]
    font_size = round(width * preset["scale"] * CAPTION_SIZES[size])
    max_width = round(width * .76)
    # Shrink crowded phrases rather than dropping words or cropping their accents.
    while True:
        font = _font(font_size, bold=preset["bold"])
        lines = _caption_wrap(text, font, max_width)
        line_height = round(font_size * 1.25)
        if (len(lines) <= 3 and len(lines) * line_height <= height * .24) or font_size <= max(12, round(width * .021)):
            break
        font_size -= 1
    padding_x, padding_y = round(width * .023), round(width * .017)
    # Keep captions clear of the top header, bottom description, and right controls.
    center_x = round(width * .47)
    block_height = len(lines) * line_height
    top = round(height * CAPTION_POSITIONS[position] - block_height / 2)
    top = max(round(height * .105) + padding_y, min(top, round(height * .81) - block_height - padding_y))
    canvas = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(canvas)
    text_width = max(draw.textlength(line, font=font) for line in lines)
    box = (round(center_x - text_width / 2 - padding_x), top - padding_y,
           round(center_x + text_width / 2 + padding_x), top + block_height + padding_y)
    radius = max(4, round(width * .017))
    stroke = max(1, round(width / 360))
    if style in {"bold", "clean"}:
        draw.rounded_rectangle(box, radius=radius, fill=(10, 13, 18, 215 if style == "clean" else 195))
    if style in {"outline", "neon", "minimal"}:
        shadow = Image.new("RGBA", (width, height))
        shadow_draw = ImageDraw.Draw(shadow)
        for row, line in enumerate(lines):
            shadow_draw.text((center_x, top + row * line_height + stroke), line, font=font, anchor="mt",
                             fill=(50, 240, 166, 170) if style == "neon" else (0, 0, 0, 255),
                             stroke_width=stroke * (2 if style == "neon" else 1),
                             stroke_fill=(50, 240, 166, 100) if style == "neon" else (0, 0, 0, 255))
        canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(stroke * (2.5 if style == "neon" else 1.5))))
        draw = ImageDraw.Draw(canvas)
    for row, line in enumerate(lines):
        y = top + row * line_height
        if style == "boxed":
            line_width = draw.textlength(line, font=font)
            draw.rounded_rectangle((round(center_x - line_width / 2 - padding_x), y - stroke * 2,
                                    round(center_x + line_width / 2 + padding_x), y + line_height - stroke),
                                   radius=max(3, radius // 2), fill=(255, 255, 255, 245))
        draw.text((center_x, y), line, font=font, anchor="mt", fill=preset["color"],
                  stroke_width=0 if style == "boxed" else stroke * (2 if style == "outline" else 1),
                  stroke_fill=(0, 0, 0, 255))
    return canvas


def _caption_track(work: Path, cues: list[dict], width: int, height: int, duration: float, style: str, size: str = "medium", position: str = "lower") -> Path:
    blank = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    blank.save(work / "blank.png")
    timeline = []
    cursor = 0.0
    for index, cue in enumerate(cues):
        if cue["start"] > cursor:
            timeline.append(("blank.png", cue["start"] - cursor))
        canvas = _caption_image(cue["text"], width, height, style, size, position)
        filename = f"caption-{index:04}.png"
        canvas.save(work / filename)
        timeline.append((filename, cue["end"] - cue["start"]))
        cursor = cue["end"]
    if cursor < duration:
        timeline.append(("blank.png", duration - cursor))
    if not timeline:
        timeline = [("blank.png", duration)]
    manifest = work / "captions.ffconcat"
    manifest.write_text("ffconcat version 1.0\n" + "".join(f"file '{filename}'\nduration {seconds:.6f}\n" for filename, seconds in timeline) + "file 'blank.png'\n", encoding="utf-8")
    return manifest


def _frame_filter(width: int, height: int, fit: str, position: float, zoom: float = 1.0, vertical_position: float = 50) -> str:
    if fit == "blur":
        return f"[0:v]split=2[back][front];[back]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=20:2[bg];[front]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30[base]"
    scaled_width = max(width, round(width * zoom / 2) * 2)
    scaled_height = max(height, round(height * zoom / 2) * 2)
    return f"[0:v]scale={scaled_width}:{scaled_height}:force_original_aspect_ratio=increase,crop={width}:{height}:x=(iw-ow)*{position / 100:.7f}:y=(ih-oh)*{vertical_position / 100:.7f},setsar=1,fps=30[base]"


def _thumbnail(folder: Path, work: Path, source: Path, start: float, duration: float, title: str, width: int, height: int, position: float, fit: str, output: Path, zoom: float = 1.0, vertical_position: float = 50) -> None:
    still = work / "frame.png"
    _run([_require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start + min(1, duration / 3)), "-i", source.name, "-frames:v", "1", str(still)], folder, timeout=60, failure="The thumbnail frame could not be extracted.")
    frame = Image.open(still).convert("RGB")
    if fit == "blur":
        canvas = ImageOps.fit(frame, (width, height)).filter(ImageFilter.GaussianBlur(width / 40))
        foreground = ImageOps.contain(frame, (width, height))
        canvas.paste(foreground, ((width - foreground.width) // 2, (height - foreground.height) // 2))
    else:
        from smart_framing import source_crop_box
        canvas = frame.crop(source_crop_box(frame.width, frame.height, width, height, zoom, position, vertical_position)).resize((width, height), Image.Resampling.LANCZOS)
    canvas = canvas.convert("RGBA")
    shade = Image.new("RGBA", (width, height))
    shade_draw = ImageDraw.Draw(shade)
    for y in range(height):
        alpha = int(225 * max(0, (y / height - .34) / .66))
        shade_draw.line((0, y, width, y), fill=(5, 10, 20, alpha))
    canvas = Image.alpha_composite(canvas, shade)
    draw = ImageDraw.Draw(canvas)
    font = _font(round(width * .085))
    lines = _wrap(title, font, round(width * .82), 4)
    line_height = round(font.size * 1.13)
    top = round(height * .70 - len(lines) * line_height / 2)
    draw.rounded_rectangle((round(width * .09), top - round(width * .07), round(width * .25), top - round(width * .05)), radius=4, fill="#FFE266")
    for index, line in enumerate(lines):
        draw.text((round(width * .09), top + index * line_height), line, font=font, fill="white", stroke_width=1, stroke_fill="#131723", anchor="lt")
    canvas.convert("RGB").save(output, format="PNG")


def render_clip(folder: Path, project: dict, clip: dict, options: dict, progress: Progress) -> dict:
    folder = Path(folder).resolve()
    source_name = str(project.get("source", ""))
    if Path(source_name).name != source_name or not source_name:
        raise VideoError("The project's source filename is invalid.")
    source = folder / source_name
    if not source.is_file():
        raise VideoError("The original video is missing. Import it again.")
    clip_id = str(clip.get("id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", clip_id):
        raise VideoError("The clip identifier is invalid.")
    duration = _number(project.get("duration"), 0, .1, MAX_DURATION, "Source duration")
    start = _number(options.get("start", clip.get("start")), 0, 0, duration, "Start time")
    end = _number(options.get("end", clip.get("end")), duration, 0, duration, "End time")
    if not 1 <= end - start <= 90.01:
        raise VideoError("Export clips must be between 1 and 90 seconds long.")
    quality = str(options.get("quality", "720")).replace("p", "")
    if quality not in {"720", "1080"}:
        raise VideoError("Choose 720p or 1080p export quality.")
    width = int(quality)
    height = width * 16 // 9
    fit = str(options.get("fit", "crop"))
    if fit not in {"crop", "blur"}:
        raise VideoError("Choose center crop or blurred background.")
    position = _number(options.get("position", clip.get("position")), 50, 0, 100, "Crop position")
    vertical_position = _number(options.get("vertical_position", clip.get("vertical_position")), 50, 0, 100, "Vertical crop position")
    from smart_framing import ZOOM_MODES, ZOOM_PRESETS, suggest_framing
    zoom_mode = str(options.get("zoom_mode", clip.get("zoom_mode", "wide")))
    if zoom_mode not in ZOOM_MODES:
        raise VideoError("Choose automatic, wide, medium, close or manual zoom.")
    zoom = _number(options.get("zoom", clip.get("zoom")), 1.0, 1.0, 2.0, "Zoom")
    captions = options.get("captions", True)
    if not isinstance(captions, bool):
        raise VideoError("Captions must be enabled or disabled.")
    style = str(options.get("caption_style", "bold"))
    if style not in CAPTION_STYLES:
        raise VideoError("Choose one of the six caption styles.")
    caption_size = str(options.get("caption_size", "medium"))
    if caption_size not in CAPTION_SIZES:
        raise VideoError("Choose small, medium or large captions.")
    caption_position = str(options.get("caption_position", "lower"))
    if caption_position not in CAPTION_POSITIONS:
        raise VideoError("Choose lower, middle or top caption placement.")
    title = re.sub(r"\s+", " ", str(options.get("title", clip.get("title", "Your next short")))).strip()[:80] or "Your next short"
    work = folder / ("render-" + uuid.uuid4().hex[:12])
    work.mkdir()
    video = folder / f"{clip_id}.mp4"
    thumbnail = folder / f"{clip_id}-thumbnail.png"
    subtitles = folder / f"{clip_id}.srt"
    cues = _clip_cues(project.get("segments", []), start, end)
    try:
        framing = {"zoom": ZOOM_PRESETS.get(zoom_mode, zoom if zoom_mode == "manual" else 1.0),
                   "position": position, "vertical_position": vertical_position, "fit": fit,
                   "reason": "Manual zoom and position." if zoom_mode == "manual" else f"{zoom_mode.capitalize()} framing preset."}
        if fit == "blur":
            framing.update(zoom=1.0, reason="Blurred background keeps the full video; zoom is disabled.")
        elif zoom_mode == "auto":
            progress("render", 4, "Checking three video frames for a safe automatic crop…")
            framing = suggest_framing(source, start, end, _require("ffmpeg"), work, position, vertical_position)
        progress("render", 8, "Preparing the vertical crop and timed captions…")
        _write_srt(work / "captions.srt", cues)
        args = [_require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-threads", "2", "-ss", f"{start:.3f}", "-i", source.name]
        filters = _frame_filter(width, height, framing["fit"], framing["position"], framing["zoom"], framing["vertical_position"])
        if captions and cues:
            manifest = _caption_track(work, cues, width, height, end - start, style, caption_size, caption_position)
            args += ["-f", "concat", "-safe", "0", "-i", str(manifest)]
            filters += ";[1:v]format=rgba,fps=30[sub];[base][sub]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[out]"
        else:
            filters += ";[base]format=yuv420p[out]"
        args += ["-filter_complex_threads", "1", "-filter_complex", filters, "-map", "[out]", "-map", "0:a?", "-t", f"{end - start:.3f}", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "22", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(work / "video.mp4")]
        progress("render", 25, "Encoding your vertical video with local FFmpeg…")
        _run(args, folder, timeout=1800, failure="The clip could not be rendered. Try 720p quality or a shorter selection.")
        progress("thumbnail", 88, "Creating a portrait cover from your video…")
        _thumbnail(folder, work, source, start, end - start, title, width, height, framing["position"], framing["fit"], work / "thumbnail.png", framing["zoom"], framing["vertical_position"])
        _probe(work / "video.mp4")
        os.replace(work / "video.mp4", video)
        os.replace(work / "thumbnail.png", thumbnail)
        os.replace(work / "captions.srt", subtitles)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    progress("ready", 100, "Your video, caption file and thumbnail are ready to download.")
    return {"video": video.name, "thumbnail": thumbnail.name, "subtitles": subtitles.name, "status": "ready", "start": round(start, 3), "end": round(end, 3), "title": title, "quality": int(quality), "fit": fit, "position": position, "vertical_position": vertical_position, "zoom_mode": zoom_mode, "zoom": zoom, "effective_fit": framing["fit"], "effective_zoom": framing["zoom"], "effective_position": framing["position"], "effective_vertical_position": framing["vertical_position"], "framing_reason": framing["reason"], "framing_detector": framing.get("detector"), "framing_detected_frames": framing.get("detected_frames", 0), "captions": captions, "caption_style": style, "caption_size": caption_size, "caption_position": caption_position, "rendered_at": time.time()}


def _metadata_only(folder: Path, url: str) -> dict:
    args = [_require('yt-dlp'), '--ignore-config', '--no-playlist', '--no-progress', '--socket-timeout', '20', '--retries', '1']
    if _binary('deno'):
        args += ['--js-runtimes', 'deno:' + _binary('deno')]
    elif _binary('node'):
        args += ['--js-runtimes', 'node:' + _binary('node')]
    raw = _run(args + ['--dump-single-json', '--skip-download', url], folder, timeout=120, failure='Could not read the video title and duration from YouTube.')
    info = json.loads(raw)
    if info.get('is_live') or info.get('live_status') == 'is_live':
        raise VideoError('Choose a finished video rather than a live broadcast.')
    _number(info.get('duration'), 0, 1, MAX_DURATION, 'Video duration')
    return info


def analyze_api_project(folder: Path, request: dict, progress: Progress) -> dict:
    from apify_transcript import ApifyError, DEFAULT_ACTOR, fetch_transcript
    from connections import OWNED_ACTOR
    from ai_ranker import rank_with_ai
    config = request.get('_api_config') or {}
    pipeline = request.get('pipeline', 'apify_ai')
    if pipeline not in {'local_ai', 'apify_ai'}:
        raise VideoError('Choose a supported AI analysis workflow.')
    key = config.get('ai_api_key')
    if not isinstance(key, str) or not key.strip() or len(key) > 1000 or any(ord(character) <= 32 for character in key.strip()):
        raise VideoError('Connect your AI provider in Connections before starting AI analysis.')
    if pipeline == 'apify_ai' and not config.get('apify_token'):
        raise VideoError('Connect Apify in Connections before starting Apify analysis.')
    url = _youtube_url(request.get('url', ''))
    progress('metadata', 3, 'Reading the video details without downloading the video…')
    info = _metadata_only(folder, url)
    duration = float(info['duration'])
    language = request.get('language', 'auto')
    if language == 'auto':
        language = str(info.get('language') or '').split('-')[0]
        if not language:
            raise VideoError("Choose the video's caption language before extracting its transcript.")
    cached_file = folder / ('local-transcript.json' if pipeline == 'local_ai' else 'apify-transcript.json')
    transcript = None
    if cached_file.is_file():
        cached = json.loads(cached_file.read_text())
        if cached.get('url') == url and cached.get('language') == language:
            transcript = cached['result']
    if transcript is None:
        if pipeline == 'local_ai':
            from local_transcript import fetch_transcript as fetch_local_transcript
            transcript = fetch_local_transcript(url, language, progress)
        else:
            run_config = dict(config)
            if request.get('apify_run_id'):
                run_config['apify_run_id'] = request['apify_run_id']
            try:
                transcript = fetch_transcript(url, language, run_config, progress)
            except ApifyError as exc:
                actor = str(run_config.get('apify_actor') or DEFAULT_ACTOR).replace('~', '/')
                if actor != OWNED_ACTOR or exc.code != 'REQUEST_BLOCKED':
                    raise
                from local_transcript import fetch_transcript as fetch_local_transcript
                try:
                    transcript = fetch_local_transcript(url, language, progress, apify_blocked=True)
                except Exception as local_error:
                    local_error.run_id = exc.run_id
                    raise
                transcript['apify_run_id'] = exc.run_id
        cached_file.write_text(json.dumps({'url': url, 'language': language, 'result': transcript}, ensure_ascii=False), encoding='utf-8')
    segments = _clean_segments(transcript['segments'], duration)
    if not segments:
        raise VideoError('Caption extraction did not return a usable timestamped transcript.')
    _write_srt(folder / 'transcript.srt', segments)
    (folder / 'transcript.txt').write_text('\n'.join(s['text'] for s in segments), encoding='utf-8')
    (folder / 'transcript.json').write_text(json.dumps({'segments': segments, 'language': language}, ensure_ascii=False), encoding='utf-8')
    progress('ai', 70, f"{config.get('ai_provider', 'Gemini').title()} is checking hooks, context and complete thoughts…")
    result = rank_with_ai(segments, duration, {**request, 'language': language}, config)
    progress('ready', 100, 'AI suggestions are ready. The source video downloads when you render your first clip.')
    return {**result, 'title': str(info.get('title') or 'YouTube video')[:200], 'duration': duration, 'language': language, 'source': None, 'source_url': url, 'segments': segments, 'pipeline': pipeline, 'transcript': 'transcript.txt', 'transcript_srt': 'transcript.srt', 'transcript_origin': transcript.get('transcript_origin', 'Local YouTube captions' if pipeline == 'local_ai' else 'Apify'), 'apify_run_id': transcript.get('apify_run_id')}


def ensure_media(folder: Path, project: dict, progress: Progress) -> dict:
    """Download video only after a clip is selected, reusing matching imports."""
    existing = str(project.get('source') or '')
    if existing and Path(existing).name == existing and (folder / existing).is_file():
        return {}
    url = _youtube_url(project.get('source_url', ''))
    for metadata in folder.parent.glob('*/project.json'):
        if metadata.parent == folder:
            continue
        try:
            other = json.loads(metadata.read_text())
            name = str(other.get('source') or '')
            source = metadata.parent / name
            if other.get('source_url') == url and name and Path(name).name == name and source.is_file() and not source.is_symlink():
                progress('import', 5, 'Reusing this video from your local workspace…')
                target = folder / ('source' + source.suffix)
                shutil.copyfile(source, target)
                media = _probe(target)
                return {'source': target.name, 'duration': media['duration'], 'width': media['width'], 'height': media['height']}
        except (OSError, ValueError):
            continue
    source, _ = _download(folder, url, str(project.get('language') or 'auto'), progress)
    media = _probe(source)
    source, media = _normalize_source(folder, source, media, progress)
    return {'source': source.name, 'duration': media['duration'], 'width': media['width'], 'height': media['height']}
