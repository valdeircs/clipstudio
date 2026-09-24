#!/usr/bin/env python3
"""ClipStudio's local-only HTTP server. No web framework or paid service required."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib
import json
import math
import mimetypes
import os
import re
import shutil
import socket
import sys
import threading
import traceback
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DATA_DIR = APP_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
UPLOADS_DIR = DATA_DIR / "uploads"
MAX_UPLOAD = 2 * 1024 * 1024 * 1024
MAX_JSON = 1024 * 1024
LOCK = threading.RLock()
WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clipstudio")
ACTIVE_STATES = {"queued", "analyzing", "processing", "rendering"}
MEDIA_TYPES = {
    ".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".mp3", ".wav",
    ".m4a", ".jpg", ".jpeg", ".png", ".webp", ".srt", ".vtt", ".ass", ".txt",
}
UPLOAD_TYPES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SECRET_FIELDS = {"apify_token", "ai_api_key", "gemini_api_key", "openai_api_key", "_api_config"}
REQUEST_FIELDS = {"url", "length", "count", "language", "pipeline", "message_goal", "content_context"}
MESSAGE_GOALS = {"balanced", "inspiring", "reflective", "encouraging", "teaching", "testimony"}
CONTENT_CONTEXTS = {"auto", "church"}
MIME_OVERRIDES = {".srt": "text/plain; charset=utf-8", ".vtt": "text/vtt; charset=utf-8", ".ass": "text/plain; charset=utf-8"}


class APIError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def project_folder(project_id: str) -> Path:
    if not ID_PATTERN.fullmatch(project_id):
        raise APIError("Project not found.", 404)
    return PROJECTS_DIR / project_id


def read_project(project_id: str) -> dict:
    with LOCK:
        path = project_folder(project_id) / "project.json"
        if not path.is_file():
            raise APIError("Project not found.", 404)
        return json.loads(path.read_text(encoding="utf-8"))


def update_project(project_id: str, change) -> dict:
    with LOCK:
        project = read_project(project_id)
        change(project)
        project["updated_at"] = now()
        atomic_json(project_folder(project_id) / "project.json", project)
        return copy.deepcopy(project)


def public_value(value):
    """Keep internal source paths and worker request details out of HTTP responses."""
    if isinstance(value, dict):
        return {key: public_value(item) for key, item in value.items()
                if not key.startswith("_") and key not in SECRET_FIELDS | {"source_file", "transcript_file", "metadata_file"}}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str) and (value.startswith(str(APP_DIR)) or value.startswith("/Users/") or value.startswith("/private/")):
        return Path(value).name
    return value


def error_message(exc: Exception, fallback: str) -> str:
    if isinstance(exc, ImportError):
        return "A required Python package is missing from this Python environment. Open Launch ClipStudio.command or follow README.md to install the dependencies."
    # Subprocess stderr can contain local paths or cookies. Only explicitly
    # user-facing engine exceptions are allowed through, with paths removed.
    message = str(exc).strip()
    if isinstance(exc, (ValueError, FileNotFoundError, RuntimeError)) and message and len(message) <= 450:
        message = re.sub(r"(?:/Users/|/private/|/tmp/)[^\s\n]+", "[local file]", message)
        if not any(word in message.lower() for word in ("cookie", "authorization", "token=", "password")):
            return message
    return fallback


def redact(value, secrets=()):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, dict):
        return {key: redact(item, secrets) for key, item in value.items() if key not in SECRET_FIELDS}
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    return value


def log_failure(project_id: str, operation: str, secrets=()) -> None:
    """Retain useful local diagnostics without making tracebacks downloadable."""
    try:
        with (project_folder(project_id) / "diagnostic.log").open("a", encoding="utf-8") as output:
            output.write(f"\n[{now()}] {operation}\n{redact(traceback.format_exc(), secrets)}\n")
    except OSError:
        pass


def engine():
    return importlib.import_module("engine")


def connections():
    return importlib.import_module("connections")


def api_config(config=None, *, pipeline: str) -> dict:
    snapshot = copy.deepcopy(config if config is not None else connections().load(resolve_apify=pipeline == "apify_ai"))
    if not snapshot.get("ai_api_key"):
        raise APIError("Connect your AI provider in Connections before starting AI analysis.", 409)
    if pipeline == "apify_ai":
        if not snapshot.get("apify_token"):
            raise APIError("Connect Apify in Connections before starting Apify analysis.", 409)
        if connections().requires_usage_consent(snapshot) and snapshot.get("apify_paid_allowed") is not True:
            raise APIError("Enable the capped Apify usage option in Connections before API analysis.", 409)
    else:
        snapshot.pop("apify_token", None)
    snapshot["pipeline"] = pipeline
    return snapshot


def analysis_settings(body: dict) -> dict:
    length = finite_number(body.get("length", 30), "Clip length")
    count = finite_number(body.get("count", 3), "Clip count")
    if not 15 <= length <= 90 or not 1 <= count <= 8 or count != int(count):
        raise APIError("Choose a clip length from 15 to 90 seconds and 1 to 8 clips.")
    language = body.get("language", "auto")
    if not isinstance(language, str) or not re.fullmatch(r"(?:auto|[A-Za-z]{2,3}(?:[-_][A-Za-z]{2,4})?)", language):
        raise APIError("Choose a valid transcription language.")
    return {"length": length, "count": int(count), "language": language, **selection_preferences(body)}


def selection_preferences(body: dict) -> dict:
    goal = body.get("message_goal", "balanced")
    context = body.get("content_context", "auto")
    if not isinstance(goal, str) or goal not in MESSAGE_GOALS:
        raise APIError("Choose a supported message goal.")
    if not isinstance(context, str) or context not in CONTENT_CONTEXTS:
        raise APIError("Choose automatic context or Church / faith.")
    return {"message_goal": goal, "content_context": context}


def capabilities() -> dict:
    try:
        return engine().capabilities()
    except ImportError:
        return {"ready": False, "error": "A local dependency is missing. Follow the setup steps in README.md.", "ffmpeg": bool(shutil.which("ffmpeg"))}
    except Exception:
        return {"ready": False, "error": "Could not check the local video tools. Restart the app and check README.md."}


def progress_callback(project_id: str, clip_id: str | None = None, secrets=()):
    def progress(stage: str, percent: int, message: str) -> None:
        def change(project):
            project["progress"] = {"stage": redact(str(stage), secrets), "percent": max(0, min(100, int(percent))), "message": redact(str(message), secrets)}
            if clip_id:
                project["progress"]["clip_id"] = clip_id
        update_project(project_id, change)
    return progress


def analyze_job(project_id: str, request: dict) -> None:
    snapshot = request.get("_api_config", {})
    secrets = tuple(value for key, value in snapshot.items() if key in SECRET_FIELDS and isinstance(value, str))
    update_project(project_id, lambda p: p.update(status="analyzing", error=None))
    try:
        def transcript_ready(metadata):
            # Extraction succeeds independently of AI ranking. Keep its result
            # reviewable even when the provider rejects a later selection.
            fields = {"title", "duration", "language", "source_url", "segments", "pipeline",
                      "transcript", "transcript_srt", "transcript_origin", "transcript_precision", "transcript_precision_note", "apify_run_id"}
            safe = redact({key: value for key, value in metadata.items() if key in fields}, secrets)
            update_project(project_id, lambda p: p.update(safe))
        observed_request = {**request, "_transcript_observer": transcript_ready,
                            "_response_observer": lambda response: atomic_json(
                                project_folder(project_id) / "ai-selection-response.json", redact(response, secrets))}
        result = engine().analyze_project(project_folder(project_id), observed_request, progress_callback(project_id, secrets=secrets))
        if not isinstance(result, dict):
            raise RuntimeError("The video analysis did not return a project.")
        def complete(project):
            old_source = project.get("source")
            safe_result = redact(result, secrets)
            project.update({key: value for key, value in safe_result.items() if key not in {"id", "created_at"} and not key.startswith("_")})
            if not project.get("source") and old_source and Path(old_source).name == old_source and (project_folder(project_id) / old_source).is_file():
                project["source"] = old_source
            for index, clip in enumerate(project.get("clips", [])):
                clip.setdefault("id", f"clip-{index + 1}")
                clip.setdefault("status", "suggested")
            project.update(status="ready", error=None, progress={"stage": "ready", "percent": 100, "message": "Your suggested clips are ready to review."})
        update_project(project_id, complete)
    except Exception as exc:
        log_failure(project_id, "Analysis", secrets)
        message = redact(error_message(exc, "Analysis failed. Check the video and installed tools, then try analysis again."), secrets)
        def failed(project):
            detail = message + " Your transcript is saved and can be reviewed or used to select clips again." if project.get("segments") else message
            project.update(status="error", error=detail, progress={"stage": "error", "percent": 0, "message": detail})
            run_id = getattr(exc, "run_id", None)
            if isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
                project["apify_run_id"] = run_id
        update_project(project_id, failed)


def render_job(project_id: str, clip_ids: list[str], options: dict) -> None:
    failures = []
    update_project(project_id, lambda p: p.update(status="rendering", error=None))
    for clip_id in clip_ids:
        project = read_project(project_id)
        clip = next(item for item in project["clips"] if item["id"] == clip_id)
        def started(p):
            target = next(item for item in p["clips"] if item["id"] == clip_id)
            target.update(status="rendering", error=None)
        update_project(project_id, started)
        try:
            if not project.get("source"):
                media = engine().ensure_media(project_folder(project_id), project, progress_callback(project_id, clip_id))
                if not isinstance(media, dict) or not media.get("source"):
                    raise RuntimeError("The source video could not be prepared for export.")
                update_project(project_id, lambda p: p.update({key: value for key, value in media.items() if key in {"source", "duration", "width", "height"}}))
                project = read_project(project_id)
                clip = next(item for item in project["clips"] if item["id"] == clip_id)
            effective_options = {**clip.get("render_options", {}), **options}
            result = engine().render_clip(project_folder(project_id), project, clip, effective_options, progress_callback(project_id, clip_id))
            if not isinstance(result, dict):
                raise RuntimeError("The renderer did not return a clip.")
            def complete(p):
                target = next(item for item in p["clips"] if item["id"] == clip_id)
                target.update({key: value for key, value in result.items() if key != "id"})
                target.update(status="ready", error=None, reviewed=True, rendered_at=now())
            update_project(project_id, complete)
        except Exception as exc:
            log_failure(project_id, "Render")
            message = error_message(exc, "Rendering failed. Check the selected range and local video tools, then try again.")
            failures.append(message)
            def failed(p):
                target = next(item for item in p["clips"] if item["id"] == clip_id)
                target.update(status="error", error=message)
            update_project(project_id, failed)
    def finished(project):
        # A render error must leave a project reviewable and retryable.
        project["status"] = "ready"
        project["error"] = failures[0] if failures else None
        project["progress"] = {"stage": "error" if failures else "ready", "percent": 100,
                               "message": f"{len(failures)} clip(s) could not be exported. You can retry them." if failures else "Your exports are ready to download."}
    update_project(project_id, finished)


def reselect_ai_job(project_id: str, request: dict) -> None:
    """Review saved captions once, committing new clips only after AI succeeds."""
    snapshot = request["_api_config"]
    secrets = tuple(value for key, value in snapshot.items() if key in SECRET_FIELDS and isinstance(value, str))
    update_project(project_id, lambda p: p.update(status="analyzing", error=None))
    try:
        from ai_ranker import rank_with_ai
        project = read_project(project_id)
        progress_callback(project_id, secrets=secrets)("ai", 40, "Gemini is choosing complete moments between 30 and 90 seconds from your saved transcript…")
        ai_request = {**request, "_response_observer": lambda response: atomic_json(
            project_folder(project_id) / "ai-selection-response.json", redact(response, secrets))}
        result = rank_with_ai(project["segments"], project["duration"], ai_request, snapshot)
        if not isinstance(result, dict) or not result.get("clips"):
            raise RuntimeError("AI returned no usable moments. Your existing clips have been kept.")
        safe_result = redact(result, secrets)
        revision = uuid.uuid4().hex[:10]
        for index, clip in enumerate(safe_result["clips"], 1):
            # Each revision has its own export paths, preserving prior renders.
            clip["id"] = f"ai-{revision}-{index}"
        def complete(current):
            previous = current.get("clips", [])
            if previous:
                current.setdefault("clip_history", []).append({
                    "saved_at": now(), "clips": copy.deepcopy(previous),
                    "analysis_method": current.get("analysis_method"),
                    "ai_provider": current.get("ai_provider"), "model": current.get("model"),
                })
            current.update({key: value for key, value in safe_result.items()
                            if key in {"clips", "ai_provider", "model", "analysis_method", "usage", "duration_mode", "min_clip_seconds", "max_clip_seconds", "coverage_note", "selection_warning", "message_goal", "content_context"}})
            preferences = selection_preferences(request)
            current["settings"] = {**current.get("settings", {}), "count": request["count"], "duration_mode": "ai", **preferences}
            if current.get("source_url"):
                current["pipeline"] = "local_ai"
                current.setdefault("_request", {}).update(pipeline="local_ai", count=request["count"], **preferences)
            current.pop("_reselect_pending", None)
            current.update(status="ready", error=None, progress={"stage": "ready", "percent": 100,
                           "message": "AI chose complete 30–90 second moments. Review each opening and ending before export."})
        update_project(project_id, complete)
    except Exception as exc:
        log_failure(project_id, "AI reselection", secrets)
        detail = redact(error_message(exc, "AI could not select new moments."), secrets)
        def failed(project):
            project.pop("_reselect_pending", None)
            message = detail + " Your existing clips and exports have been kept."
            project.update(status="ready" if project.get("clips") else "error", error=message,
                           progress={"stage": "error", "percent": 0, "message": message})
        update_project(project_id, failed)


def finite_number(value, label: str) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError()
        number = float(value)
        if not math.isfinite(number):
            raise ValueError()
        return number
    except (TypeError, ValueError):
        raise APIError(f"{label} must be a valid number.")


def render_options(body: dict) -> dict:
    if not isinstance(body, dict):
        raise APIError("Export options must be a JSON object.")
    result = {}
    for key in ("start", "end"):
        if key in body:
            result[key] = finite_number(body[key], key.capitalize())
    if "title" in body:
        if not isinstance(body["title"], str):
            raise APIError("Title must be text.")
        result["title"] = body["title"].strip()[:80]
    if "fit" in body:
        if not isinstance(body["fit"], str) or body["fit"] not in {"crop", "blur"}:
            raise APIError("Choose crop or blurred background.")
        result["fit"] = body["fit"]
    if "position" in body:
        result["position"] = max(0, min(100, finite_number(body["position"], "Position")))
    if "vertical_position" in body:
        result["vertical_position"] = max(0, min(100, finite_number(body["vertical_position"], "Vertical position")))
    if "zoom_mode" in body:
        if not isinstance(body["zoom_mode"], str) or body["zoom_mode"] not in {"auto", "wide", "medium", "close", "manual"}:
            raise APIError("Choose automatic framing, a zoom preset, or manual zoom.")
        result["zoom_mode"] = body["zoom_mode"]
    if "zoom" in body:
        zoom = finite_number(body["zoom"], "Zoom")
        if not 1 <= zoom <= 2:
            raise APIError("Choose a zoom from 1× to 2×.")
        result["zoom"] = zoom
    if "captions" in body:
        if not isinstance(body["captions"], bool):
            raise APIError("Captions must be true or false.")
        result["captions"] = body["captions"]
    if "caption_style" in body:
        if not isinstance(body["caption_style"], str) or body["caption_style"] not in {"bold", "clean", "boxed", "outline", "neon", "minimal"}:
            raise APIError("Choose a supported caption style.")
        result["caption_style"] = body["caption_style"]
    for field, allowed, message in (
        ("caption_size", {"small", "medium", "large"}, "Choose small, medium or large captions."),
        ("caption_position", {"lower", "middle", "top"}, "Choose lower, middle or top caption placement."),
    ):
        if field in body:
            if not isinstance(body[field], str) or body[field] not in allowed:
                raise APIError(message)
            result[field] = body[field]
    if "quality" in body:
        if not isinstance(body["quality"], (int, str)) or body["quality"] not in {720, 1080, "720", "1080"}:
            raise APIError("Choose 720p or 1080p.")
        result["quality"] = int(body["quality"])
    return result


class LocalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "ClipStudio/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        # Query strings and uploaded file names may be private.
        pass

    def send_json(self, data, status: int = 200) -> None:
        payload = json.dumps(public_value(data), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.security_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")

    def check_host(self, mutation: bool = False) -> None:
        port = self.server.server_port
        host = self.headers.get("Host", "").lower()
        allowed = {f"localhost:{port}", f"127.0.0.1:{port}"}
        if port == 80:
            allowed.update({"localhost", "127.0.0.1"})
        if host not in allowed:
            raise APIError("This app only accepts requests from its local address.", 403)
        if mutation:
            if self.headers.get("X-ClipStudio") != "1":
                raise APIError("Missing local app request header. Reload ClipStudio and try again.", 403)
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{host}":
                raise APIError("Requests from another website are not allowed.", 403)
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise APIError("Requests from another website are not allowed.", 403)

    def content_length(self, maximum: int) -> int:
        if self.headers.get("Transfer-Encoding"):
            raise APIError("Chunked uploads are not supported. Please send a file with a known size.", 411)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError("Invalid content length.")
        if length < 0 or length > maximum:
            raise APIError("The file is too large. The local upload limit is 2 GB." if maximum == MAX_UPLOAD else "Request too large.", 413)
        return length

    def read_json(self) -> dict:
        length = self.content_length(MAX_JSON)
        if length == 0:
            return {}
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
            raise APIError("Send this request as JSON.", 415)
        data = self.rfile.read(length)
        if len(data) != length:
            raise APIError("Incomplete request body.")
        try:
            body = json.loads(data)
        except (ValueError, UnicodeError):
            raise APIError("Invalid JSON request.")
        if not isinstance(body, dict):
            raise APIError("Request must be a JSON object.")
        return body

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        try:
            self.check_host()
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if path == "/api/status":
                self.send_json({"app": "ClipStudio", "version": "1.0", "local": True, "capabilities": capabilities()})
            elif path == "/api/connections":
                self.send_json(connections().public_status())
            elif path == "/api/projects":
                with LOCK:
                    projects = []
                    for file in PROJECTS_DIR.glob("*/project.json"):
                        try:
                            project = json.loads(file.read_text(encoding="utf-8"))
                            # Transcripts can be large; project detail loads them on demand.
                            project.pop("segments", None)
                            projects.append(project)
                        except (OSError, ValueError):
                            continue
                projects.sort(key=lambda p: p.get("created_at", ""), reverse=True)
                self.send_json({"projects": projects})
            elif match := re.fullmatch(r"/api/projects/([0-9a-f]{32})", path):
                self.send_json(read_project(match[1]))
            elif match := re.fullmatch(r"/api/projects/([0-9a-f]{32})/download", path):
                name = parse_qs(parsed.query).get("file", [""])[0]
                self.serve_media(match[1], name, download=True)
            elif match := re.fullmatch(r"/media/([0-9a-f]{32})/([^/]+)", path):
                self.serve_media(match[1], match[2], download=parse_qs(parsed.query).get("download") == ["1"])
            elif path.startswith("/api/") or path.startswith("/media/"):
                raise APIError("Not found.", 404)
            else:
                relative = "index.html" if path == "/" else path.lstrip("/")
                file = (STATIC_DIR / relative).resolve()
                if not file.is_relative_to(STATIC_DIR.resolve()) or not file.is_file():
                    raise APIError("Not found.", 404)
                self.serve_file(file)
        except APIError as exc:
            self.send_json({"error": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.send_json({"error": "The local app could not complete that request."}, 500)

    def do_POST(self) -> None:
        try:
            self.connection.settimeout(60)
            self.check_host(mutation=True)
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if path == "/api/upload":
                self.upload(parse_qs(parsed.query))
                return
            body = self.read_json()
            if path == "/api/projects":
                self.create_project(body)
            elif path == "/api/connections":
                try:
                    result = connections().save(body)
                except ValueError as exc:
                    raise APIError(str(exc))
                self.send_json(result)
            elif match := re.fullmatch(r"/api/projects/([0-9a-f]{32})/reanalyze", path):
                self.reanalyze_project(match[1], body)
            elif match := re.fullmatch(r"/api/projects/([0-9a-f]{32})/reselect-ai", path):
                self.reselect_ai(match[1], body)
            elif match := re.fullmatch(r"/api/projects/([0-9a-f]{32})/clips/([^/]+)/render", path):
                self.queue_render(match[1], [match[2]], body)
            elif match := re.fullmatch(r"/api/projects/([0-9a-f]{32})/render-all", path):
                clip_ids = body.get("clip_ids")
                if clip_ids is not None and (not isinstance(clip_ids, list) or not clip_ids or not all(isinstance(item, str) for item in clip_ids)):
                    raise APIError("Select at least one clip to export.")
                self.queue_render(match[1], clip_ids, body)
            else:
                raise APIError("Not found.", 404)
        except APIError as exc:
            self.close_connection = True
            self.send_json({"error": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self.close_connection = True
        except Exception:
            self.close_connection = True
            self.send_json({"error": "The local app could not complete that request. Check available disk space and try again."}, 500)

    def do_OPTIONS(self) -> None:
        self.close_connection = True
        self.send_json({"error": "Cross-origin requests are not allowed."}, 403)

    def upload(self, query: dict) -> None:
        name = Path(query.get("name", ["video.mp4"])[0].replace("\\", "/")).name
        extension = Path(name).suffix.lower()
        if extension not in UPLOAD_TYPES:
            raise APIError("Upload an MP4, MOV, WebM, MKV, AVI, or M4V video.")
        length = self.content_length(MAX_UPLOAD)
        if length == 0:
            raise APIError("The uploaded video is empty.")
        upload_id = uuid.uuid4().hex
        folder = UPLOADS_DIR / upload_id
        folder.mkdir(parents=True)
        target = folder / ("source" + extension)
        partial = target.with_suffix(target.suffix + ".part")
        try:
            remaining = length
            with partial.open("wb") as output:
                while remaining:
                    block = self.rfile.read(min(1024 * 1024, remaining))
                    if not block:
                        raise APIError("Upload interrupted. Please try again.")
                    output.write(block)
                    remaining -= len(block)
            partial.replace(target)
            atomic_json(folder / "upload.json", {"name": name[:250], "filename": target.name, "size": length, "created_at": now()})
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        self.send_json({"upload_id": upload_id, "name": name[:250]}, 201)

    def create_project(self, body: dict) -> None:
        settings = analysis_settings(body)
        request = {**settings, "pipeline": "local"}
        title = "New video"
        source_kind = "youtube"
        if body.get("demo") is True:
            demo = APP_DIR / "demo"
            if not (demo / "source.mp4").is_file() or not (demo / "transcript.json").is_file():
                raise APIError("The bundled demo is not available yet.", 503)
            request.update(demo=True, source_file=str(demo / "source.mp4"), transcript_file=str(demo / "transcript.json"), metadata_file=str(demo / "metadata.json"))
            title = "The art of a great short"
            source_kind = "demo"
        elif body.get("upload_id"):
            upload_id = body["upload_id"]
            if not isinstance(upload_id, str) or not ID_PATTERN.fullmatch(upload_id):
                raise APIError("Upload not found. Please upload your video again.", 404)
            folder = UPLOADS_DIR / upload_id
            metadata_file = folder / "upload.json"
            if not metadata_file.is_file():
                raise APIError("Upload not found. Please upload your video again.", 404)
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            request["source_file"] = str(folder / metadata["filename"])
            request["title"] = Path(metadata["name"]).stem
            title = request["title"]
            source_kind = "upload"
        else:
            url = body.get("url", "")
            if not isinstance(url, str) or len(url) > 3000:
                raise APIError("Paste a valid YouTube video link.")
            try:
                parsed = urlsplit(url.strip())
            except ValueError:
                raise APIError("Paste a valid YouTube video link.")
            if parsed.scheme not in {"https", "http"} or parsed.hostname not in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"} or parsed.username or parsed.password:
                raise APIError("Paste a YouTube video link, or upload a local video.")
            try:
                if parsed.port not in {None, 80, 443}:
                    raise APIError("Use a standard YouTube video link.")
            except ValueError:
                raise APIError("Paste a valid YouTube video link.")
            request["url"] = url.strip()
            config = None
            pipeline = body.get("pipeline")
            if pipeline is None:
                config = connections().load(resolve_apify=False)
                pipeline = config.get("pipeline", "local_ai")
            if not isinstance(pipeline, str) or pipeline not in {"local", "local_ai", "apify_ai"}:
                raise APIError("Choose local captions with AI, Apify with AI, or local analysis.")
            request["pipeline"] = pipeline
            if pipeline in {"local_ai", "apify_ai"}:
                request["_api_config"] = api_config(config if pipeline == "local_ai" else None, pipeline=pipeline)
        project_id = uuid.uuid4().hex
        created = now()
        project = {"id": project_id, "title": title, "status": "queued", "created_at": created, "updated_at": created,
                   "source_kind": source_kind, "demo": source_kind == "demo", "settings": settings, "pipeline": request["pipeline"],
                   "_request": {key: value for key, value in request.items() if key in REQUEST_FIELDS},
                   "progress": {"stage": "queued", "percent": 0, "message": "Waiting for the local worker…"}, "clips": [], "error": None}
        with LOCK:
            atomic_json(project_folder(project_id) / "project.json", project)
            WORKER.submit(analyze_job, project_id, request)
        self.send_json(project, 202)

    def reanalyze_project(self, project_id: str, body: dict) -> None:
        with LOCK:
            project = read_project(project_id)
            if project.get("status") in ACTIVE_STATES:
                raise APIError("This project already has a job running. Wait for it to finish.", 409)
            original = project.get("_request", {})
            pipeline = project.get("pipeline", original.get("pipeline", "local"))
            if "pipeline" in body and body["pipeline"] != pipeline:
                raise APIError("Reanalysis uses this project's existing pipeline. Create a new project to change pipelines.")
            settings = analysis_settings({**project.get("settings", {}), **{key: value for key, value in body.items() if key in {"length", "count", "language", "message_goal", "content_context"}}})
            request = {**{key: value for key, value in original.items() if key in REQUEST_FIELDS}, **settings, "pipeline": pipeline}
            source_url = request.get("url") or project.get("source_url")
            if source_url:
                request["url"] = source_url
            if pipeline in {"local_ai", "apify_ai"}:
                if not source_url:
                    raise APIError("This project's YouTube link is missing. Create the project again.")
                request["_api_config"] = api_config(pipeline=pipeline)
                if pipeline == "apify_ai" and project.get("apify_run_id"):
                    request["_api_config"]["apify_run_id"] = project["apify_run_id"]
            elif project.get("demo"):
                demo = APP_DIR / "demo"
                request.update(demo=True, source_file=str(demo / "source.mp4"), transcript_file=str(demo / "transcript.json"), metadata_file=str(demo / "metadata.json"))
            else:
                source = project.get("source")
                if source and Path(source).name == source and (project_folder(project_id) / source).is_file():
                    request.update(source_file=str(project_folder(project_id) / source), title=project.get("title", "Video"))
                    transcript = project_folder(project_id) / "transcript.json"
                    if transcript.is_file() and settings["language"] == project.get("settings", {}).get("language", "auto"):
                        request["transcript_file"] = str(transcript)
                elif not source_url:
                    raise APIError("The source video is unavailable. Upload it again to create a new project.")
            project.update(status="queued", error=None, settings=settings, updated_at=now(),
                           _request={key: value for key, value in request.items() if key in REQUEST_FIELDS},
                           progress={"stage": "queued", "percent": 0, "message": "Analysis queued for the local worker…"})
            atomic_json(project_folder(project_id) / "project.json", project)
            WORKER.submit(analyze_job, project_id, request)
        self.send_json(project, 202)

    def reselect_ai(self, project_id: str, body: dict) -> None:
        count = finite_number(body.get("count", 5), "Clip count")
        if count != int(count) or not 1 <= count <= 8:
            raise APIError("Choose from 1 to 8 AI suggestions.")
        with LOCK:
            project = read_project(project_id)
            if project.get("status") in ACTIVE_STATES:
                raise APIError("This project already has a job running. Wait for it to finish.", 409)
            if not isinstance(project.get("segments"), list) or not project["segments"]:
                raise APIError("This project needs a transcript before AI can select its moments.", 409)
            if finite_number(project.get("duration", 0), "Video duration") < 30:
                raise APIError("The source must be at least 30 seconds long for AI selection.")
            config = api_config(pipeline="local_ai")
            preferences = selection_preferences({**project.get("settings", {}), **{key: value for key, value in body.items() if key in {"message_goal", "content_context"}}})
            request = {"count": int(count), "language": project.get("language", "auto"), "_api_config": config, **preferences}
            project.update(status="queued", error=None, _reselect_pending=True, updated_at=now(),
                           progress={"stage": "queued", "percent": 0, "message": "AI review queued. Your existing clips stay saved until the new selections are ready."})
            atomic_json(project_folder(project_id) / "project.json", project)
            WORKER.submit(reselect_ai_job, project_id, request)
        self.send_json(project, 202)

    def queue_render(self, project_id: str, requested_ids: list[str] | None, body: dict) -> None:
        options = render_options(body.get("options", body))
        with LOCK:
            project = read_project(project_id)
            if project["status"] in ACTIVE_STATES:
                raise APIError("This project already has a job running. Wait for it to finish.", 409)
            clips = project.get("clips", [])
            available = {clip["id"] for clip in clips}
            ids = list(dict.fromkeys(requested_ids)) if requested_ids is not None else [clip["id"] for clip in clips]
            if not ids or any(clip_id not in available for clip_id in ids):
                raise APIError("Clip not found.", 404)
            if len(ids) > 1 and any(key in options for key in ("start", "end", "title")):
                raise APIError("Edit titles and time ranges one clip at a time.")
            for clip in clips:
                if clip["id"] not in ids:
                    continue
                start = finite_number(options.get("start", clip.get("start", 0)), "Start time")
                end = finite_number(options.get("end", clip.get("end", 0)), "End time")
                if start < 0 or end - start < 1 or end - start > 90:
                    raise APIError("Choose a clip range from 1 to 90 seconds.")
                duration = project.get("duration")
                if isinstance(duration, (int, float)) and end > duration + 0.1:
                    raise APIError("The clip cannot extend past the end of the source video.")
                clip.update(start=start, end=end, duration=round(end - start, 3), status="queued", reviewed=True, error=None)
                if "title" in options:
                    clip["title"] = options["title"]
                saved_options = {**clip.get("render_options", {}), **{key: value for key, value in options.items() if key not in {"start", "end", "title"}}}
                clip["render_options"] = saved_options
                clip.update(saved_options)
                # Old files remain on disk, but a failed new render must not be
                # presented as a successful export of the newly approved edits.
                for key in ("video", "thumbnail", "subtitles", "rendered_at"):
                    clip.pop(key, None)
            project.update(status="queued", updated_at=now(), error=None, progress={"stage": "queued", "percent": 0, "message": "Export queued for the local worker…"})
            atomic_json(project_folder(project_id) / "project.json", project)
            WORKER.submit(render_job, project_id, ids, options)
        self.send_json(project, 202)

    def serve_media(self, project_id: str, name: str, download: bool = False) -> None:
        folder = project_folder(project_id)
        if not name or name != Path(name).name or "\\" in name or Path(name).suffix.lower() not in MEDIA_TYPES:
            raise APIError("Media not found.", 404)
        file = (folder / name).resolve()
        if file.parent != folder.resolve() or not file.is_file():
            raise APIError("Media not found.", 404)
        self.serve_file(file, download=download, ranges=True)

    def serve_file(self, path: Path, download: bool = False, ranges: bool = False) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range") if ranges else None
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            valid = bool(match and (match[1] or match[2]) and size > 0)
            if valid:
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]), size - 1) if match[2] else size - 1
                else:
                    tail = int(match[2])
                    valid = tail > 0
                    start = max(0, size - tail)
                valid = valid and start < size and start <= end
            if not valid:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", MIME_OVERRIDES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.send_header("Cache-Control", "no-cache")
        if ranges:
            self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", path.name)
            self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
        self.security_headers()
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as source:
            source.seek(start)
            remaining = max(0, end - start + 1)
            while remaining:
                block = source.read(min(256 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)


def recover_interrupted_jobs() -> None:
    for path in PROJECTS_DIR.glob("*/project.json"):
        try:
            with LOCK:
                project = json.loads(path.read_text(encoding="utf-8"))
                if project.get("status") in ACTIVE_STATES:
                    for clip in project.get("clips", []):
                        if clip.get("status") in ACTIVE_STATES:
                            clip.update(status="error", error="Export interrupted when the app closed. You can retry it.")
                    has_clips = bool(project.get("clips"))
                    message = "The app closed before the job finished. Retry your export." if has_clips else "The app closed before analysis finished. Please create the project again."
                    if project.pop("_reselect_pending", False):
                        message = "AI review was interrupted. Your existing clips and exports have been kept. You can request a new AI review."
                    project.update(status="ready" if has_clips else "error", error=message, updated_at=now(), progress={"stage": "error", "percent": 0, "message": message})
                    atomic_json(path, project)
        except (OSError, ValueError):
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the free local ClipStudio app.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CLIPSTUDIO_PORT", "8765")))
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    # A directly invoked .venv/bin/python should also find its installed CLIs.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be from 1 to 65535.")
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        httpd = LocalHTTPServer(("127.0.0.1", args.port), Handler)
    except PermissionError:
        print("This environment blocked opening a local server. Run the launcher in Terminal with permission to listen on localhost.", flush=True)
        return 1
    except OSError as exc:
        import errno
        if exc.errno != errno.EADDRINUSE:
            print("ClipStudio could not open its local server. Check your network permissions and try another port.", flush=True)
            return 1
        # Reopening the launcher should show an existing instance, never kill it.
        try:
            from urllib.request import urlopen
            with urlopen(f"http://127.0.0.1:{args.port}/api/status", timeout=2) as response:
                existing = json.load(response)
            if existing.get("app") == "ClipStudio":
                print(f"ClipStudio is already running at http://127.0.0.1:{args.port}", flush=True)
                if args.open_browser:
                    webbrowser.open(f"http://127.0.0.1:{args.port}")
                return 0
        except Exception:
            pass
        print(f"Port {args.port} is in use. Try: python3 server.py --port {args.port + 1}", flush=True)
        return 1
    recover_interrupted_jobs()
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nClipStudio is running at {url}\nKeep this window open. Press Control+C to stop.\n", flush=True)
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nStopping ClipStudio…", flush=True)
    finally:
        httpd.server_close()
        WORKER.shutdown(wait=False, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
