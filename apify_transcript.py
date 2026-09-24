"""Apify captions adapter, using only Python's standard library.

The default actor requires public videos with captions. Its input and output
schema are supplied by the user's agency-shift/youtube-transcript-scraper actor.
This owned actor has no third-party actor fee; Apify compute and storage still
use the account's allowance or billing. No request is retried automatically.
"""
from __future__ import annotations

import html
import http.client
import json
import math
import re
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_ACTOR = "agency-shift/youtube-transcript-scraper"
API_BASE = "https://api.apify.com/v2"
MAX_RESPONSE_BYTES = 3 * 1024 * 1024
REQUEST_DEADLINE_SECONDS = 150
SUPPORTED_LANGUAGES = frozenset(
    "en pt es fr de it ja ko zh ru ar hi nl pl tr vi id th sv da no fi cs hu ro "
    "uk el he fa bn ta te mr ur gu kn ml pa sr hr bg sk lt lv et sl ca af sq am "
    "hy az eu be bs my ceb zh-Hans zh-Hant co eo fil fy gl ka ht ha haw hmn is "
    "ig ga jv kk km rw ku ky lo la lb mk mg ms mt mi mn ne ny or ps sm gd sn sd "
    "si so st su sw tg tt tk ug uz cy xh yi yo zu".split()
)
_RESOURCE_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_ACTOR_ID = re.compile(r"[A-Za-z0-9_-]{1,100}(?:[/~][A-Za-z0-9_-]{1,100})?\Z")
_PENDING = {"READY", "RUNNING", "TIMING-OUT", "ABORTING"}
_FAILED = {"FAILED", "TIMED-OUT", "ABORTED"}


class ApifyError(RuntimeError):
    """Safe public error; run_id can be retained for a later status check."""

    def __init__(self, message: str, run_id: str | None = None, code: str | None = None):
        super().__init__(message)
        self.run_id = run_id
        self.code = code


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # In particular, never forward the bearer credential to another host.
        return None


def _canonical_video_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if (parsed.scheme not in {"http", "https"} or parsed.username or
                parsed.password or parsed.port is not None):
            raise ValueError
        if host in {"youtu.be", "www.youtu.be"}:
            video_id = parsed.path.removeprefix("/")
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
            if parsed.path == "/watch":
                ids = parse_qs(parsed.query).get("v", [])
                video_id = ids[0] if len(ids) == 1 else ""
            else:
                match = re.fullmatch(r"/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})/?", parsed.path)
                video_id = match.group(1) if match else ""
        else:
            raise ValueError
        if not _VIDEO_ID.fullmatch(video_id):
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise ApifyError("Enter a valid YouTube video URL.") from None
    return f"https://www.youtube.com/watch?v={video_id}"


def _resource_id(value, label: str) -> str:
    if not isinstance(value, str) or not _RESOURCE_ID.fullmatch(value):
        raise ApifyError(f"Apify returned an invalid {label} identifier.")
    return value


def _request_json(opener, path: str, token: str, deadline: float, payload=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ApifyError("The Apify request timed out.")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        API_BASE + path, data=body,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json",
                 "Content-Type": "application/json", "User-Agent": "ClipStudio/1"},
        method="GET" if payload is None else "POST",
    )
    try:
        with opener.open(request, timeout=min(40, remaining)) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        code = exc.code
        exc.close()
        if code in {401, 403}:
            message = "Apify rejected the API token or its permissions. Check Apify settings."
        elif code == 402:
            message = "Apify requires available credits for this actor. Check your Apify account."
        elif code == 429:
            message = "Apify is rate limiting requests. Check the existing run before trying again."
        elif 300 <= code < 400:
            message = "Apify redirected the request. It was stopped to protect the API token."
        else:
            message = f"Apify returned HTTP {code}. Check the actor and account settings."
        raise ApifyError(message) from None
    except (URLError, OSError, http.client.HTTPException, ValueError):
        raise ApifyError("Apify could not be reached or the request timed out.") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ApifyError("The Apify response exceeded the 3 MB limit.")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ApifyError("Apify returned an unreadable response.") from None


def _run_data(response: dict) -> dict:
    if not isinstance(response, dict) or not isinstance(response.get("data"), dict):
        raise ApifyError("Apify did not return a run record.")
    return response["data"]


def _parse_dataset(items, language: str, actor: str, run_id: str) -> dict:
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        raise ApifyError("Apify did not return one video transcript. Check the actor output schema.")
    item = items[0]
    status = item.get("status")
    error = item.get("error")
    if (status is not None and (not isinstance(status, str) or status.lower() not in {"success", "succeeded", "ok"})) or error:
        code = error.get("code") if isinstance(error, dict) else None
        labels = {value.lower().replace("-", "_") for value in (status, code) if isinstance(value, str)}
        if labels & {"blocked", "request_blocked", "ip_blocked", "youtube_blocked", "bot_detected"}:
            message = "YouTube blocked the transcript request from Apify. Try a different public video or upload the source file for local analysis."
        elif labels & {"language_unavailable", "language_not_available", "no_matching_language", "no_transcript_found", "language_mismatch"}:
            message = "This video has no accessible captions in the selected language. Choose a caption language available on the video."
        elif labels & {"no_captions", "captions_disabled", "transcripts_disabled", "no_transcript", "transcript_disabled", "captions_unavailable", "empty_transcript"}:
            message = "This video has no accessible public captions. Upload the source file to transcribe it locally."
        elif "duration_limit" in labels:
            message = "This video exceeds the transcript scraper's duration limit. Choose a shorter source video."
        elif labels & {"unavailable", "video_unavailable", "private_video", "video_not_found", "age_restricted", "login_required"}:
            message = "The video is unavailable or restricted. Use a public video with accessible captions."
        elif labels & {"invalid_input", "invalid_url", "invalid_video_url", "invalid_language", "invalid_video_id", "invalid_proxy"}:
            message = "The transcript scraper rejected the video link or language. Check the project settings."
        elif labels & {"timeout", "timed_out", "request_timeout"}:
            message = "The transcript request timed out. Check the existing Apify run before trying again."
        elif "invalid_timestamps" in labels:
            message = "The transcript scraper did not return valid caption timing. Try a different public video or upload the source file."
        elif "transcript_too_large" in labels:
            message = "This video's captions exceed the transcript size limit. Choose a shorter source video."
        elif "network_error" in labels:
            message = "The transcript scraper could not reach YouTube. Check the existing Apify run before trying again."
        else:
            message = "The transcript scraper could not read this video. Check the existing Apify run for details."
        safe_code = code if isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) else None
        raise ApifyError(message, run_id, code=safe_code)
    rows = item.get("data")
    if not isinstance(rows, list) or not rows:
        raise ApifyError("Apify found no timestamped captions in the selected language.")
    segments = []
    seen = set()
    for row in rows:
        try:
            if not isinstance(row, dict) or isinstance(row.get("start"), bool) or isinstance(row.get("dur"), bool):
                raise ValueError
            start, duration = float(row["start"]), float(row["dur"])
            end = start + duration
            if not all(math.isfinite(n) for n in (start, duration, end)) or start < 0 or duration <= 0:
                raise ValueError
            if not isinstance(row.get("text"), str):
                raise ValueError
            text = " ".join(html.unescape(row["text"]).split())
            if not text:
                raise ValueError
        except (KeyError, ValueError, TypeError, OverflowError):
            raise ApifyError("Apify returned a caption with invalid text or timing.") from None
        key = (start, end, text)
        if key not in seen:
            segments.append({"start": start, "end": end, "text": text})
            seen.add(key)
    segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    transcript_end = max(segment["end"] for segment in segments)
    duration = transcript_end
    if item.get("duration") is not None:
        try:
            duration = float(item["duration"])
            if isinstance(item["duration"], bool) or not math.isfinite(duration) or duration < transcript_end:
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise ApifyError("Apify returned an invalid video duration.") from None
    result = {"segments": segments, "language": language, "duration": duration,
              "apify_run_id": run_id, "transcript_origin": f"Apify ({actor})"}
    if isinstance(item.get("title"), str) and item["title"].strip():
        result["title"] = item["title"].strip()[:1000]
    return result


def fetch_transcript(url: str, language: str, config: dict,
                     progress: Callable[[str, int, str], None]) -> dict:
    """Fetch captions once, or resume config.apify_run_id without starting a run.

    Config keys: apify_token, apify_actor (optional same-schema actor),
    apify_max_charge_usd (default 0.02), apify_run_id (optional). The caller
    must resolve the language before calling; the actor has no auto language.
    """
    video_url = _canonical_video_url(url)
    if language not in SUPPORTED_LANGUAGES:
        raise ApifyError("Choose the video's caption language before using Apify; automatic language is not supported.")
    token = config.get("apify_token", "")
    if not isinstance(token, str) or not token.strip() or any(ord(c) <= 32 or ord(c) > 126 for c in token.strip()):
        raise ApifyError("Add a valid Apify API token in Settings.")
    token = token.strip()
    actor = config.get("apify_actor") or DEFAULT_ACTOR
    if not isinstance(actor, str) or not _ACTOR_ID.fullmatch(actor):
        raise ApifyError("The Apify actor must be an actor ID or an owner/name identifier.")
    actor = actor.replace("~", "/")
    try:
        maximum_charge = float(config.get("apify_max_charge_usd", 0.02))
        if isinstance(config.get("apify_max_charge_usd"), bool) or not math.isfinite(maximum_charge) or not 0 < maximum_charge <= 1:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ApifyError("Set the Apify run charge cap above $0 and no higher than $1.") from None
    run_id = config.get("apify_run_id") or None
    if run_id is not None:
        run_id = _resource_id(run_id, "run")
    opener = build_opener(_NoRedirects())
    deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
    if actor != DEFAULT_ACTOR:
        progress("transcript", 20, "Using a custom Apify actor; it must accept videoUrl/targetLanguage and return timestamped data.")
    try:
        if run_id:
            progress("transcript", 22, f"Checking existing Apify run {run_id}; no new run is being started.")
            run = _run_data(_request_json(opener, f"/actor-runs/{run_id}", token, deadline))
            if run.get("id") != run_id:
                raise ApifyError("Apify returned a different run than requested.")
            store_id = _resource_id(run.get("defaultKeyValueStoreId"), "input storage")
            original = _request_json(opener, f"/key-value-stores/{store_id}/records/INPUT", token, deadline)
            if (not isinstance(original, dict) or _canonical_video_url(original.get("videoUrl")) != video_url or
                    original.get("targetLanguage") != language):
                raise ApifyError("The existing Apify run belongs to a different video or language.")
        else:
            progress("transcript", 22, f"Requesting captions through Apify (run charge cap ${maximum_charge:g})…")
            query = urlencode({"timeout": 120, "memory": 128, "maxTotalChargeUsd": f"{maximum_charge:.8g}", "restartOnError": "false"})
            try:
                run = _run_data(_request_json(opener, f"/actors/{actor.replace('/', '~')}/runs?{query}", token, deadline,
                                             {"videoUrl": video_url, "targetLanguage": language}))
                run_id = _resource_id(run.get("id"), "run")
            except ApifyError as exc:
                raise ApifyError(f"{exc} A run may have started. Check Apify's Runs page before retrying; no automatic retry was made.", code=exc.code) from None
            progress("transcript", 24, f"Apify run {run_id} started; waiting for captions…")
        while run.get("status") != "SUCCEEDED":
            status = run.get("status")
            if status in _FAILED:
                # The owned actor records a structured, sanitized failure before
                # marking its run failed. Read that one record for a useful error.
                dataset_id = run.get("defaultDatasetId")
                if actor == DEFAULT_ACTOR and status == "FAILED" and isinstance(dataset_id, str) and _RESOURCE_ID.fullmatch(dataset_id):
                    items = _request_json(opener, f"/datasets/{dataset_id}/items?clean=true&limit=2", token, deadline)
                    if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict) and (items[0].get("status") is not None or items[0].get("error")):
                        _parse_dataset(items, language, actor, run_id)
                raise ApifyError(f"Apify run {run_id} ended with status {status}. Check its output before starting another run.")
            if status not in _PENDING:
                raise ApifyError("Apify returned an unknown run status.")
            if time.monotonic() >= deadline:
                raise ApifyError(f"Apify run {run_id} is still pending. Check or resume this run before starting another.")
            run = _run_data(_request_json(opener, f"/actor-runs/{run_id}?waitForFinish=30", token, deadline))
            if run.get("id") != run_id:
                raise ApifyError("Apify returned a different run than requested.")
        dataset_id = _resource_id(run.get("defaultDatasetId"), "dataset")
        items = _request_json(opener, f"/datasets/{dataset_id}/items?clean=true&limit=2", token, deadline)
        result = _parse_dataset(items, language, actor, run_id)
        progress("transcript", 40, f"Apify captions are ready (run {run_id}).")
        return result
    except ApifyError as exc:
        if run_id:
            exc.run_id = run_id
            if run_id not in str(exc):
                raise ApifyError(f"{exc} Existing Apify run: {run_id}. No new run was started automatically.", run_id, code=exc.code) from None
        raise
