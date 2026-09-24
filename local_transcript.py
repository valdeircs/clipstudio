"""Free transcript-only local extraction, also used for blocked-cloud fallback.

No Apify credentials, proxies, video download, speech model, or AI call is used.
Install youtube-transcript-api==1.2.4; dependencies are imported only on use.
"""
from __future__ import annotations

import math
import re
import time
from urllib.parse import parse_qs, urlsplit

MAX_SECONDS = 7200
MAX_SEGMENTS = 30000
MAX_TEXT_CHARS = 1_000_000
REQUEST_SECONDS = 60


class LocalTranscriptError(RuntimeError):
    """Fixed messages suitable for the application, with no upstream details."""


def _select_track(tracks, language):
    target = language.lower()
    base = target.split("-")[0]
    matches = [track for track in tracks if track.language_code.lower().split("-")[0] == base]
    if not matches:
        raise LocalTranscriptError("No captions are available in the selected language on this Mac. Choose an available language or upload the video for local transcription.")
    return min(matches, key=lambda track: (track.language_code.lower() != target, bool(track.is_generated), track.language_code))


def _normalize(snippets):
    segments = []
    characters = 0
    for index, snippet in enumerate(snippets):
        if index >= MAX_SEGMENTS:
            raise LocalTranscriptError("The local transcript exceeds the caption segment limit.")
        try:
            start, duration = float(snippet.start), float(snippet.duration)
            if isinstance(snippet.start, bool) or isinstance(snippet.duration, bool) or not math.isfinite(start + duration) or start < 0 or duration <= 0 or start + duration > MAX_SECONDS:
                raise ValueError
            if not isinstance(snippet.text, str):
                raise ValueError
            text = re.sub(r"\s+", " ", snippet.text).strip()
        except (AttributeError, TypeError, ValueError, OverflowError):
            raise LocalTranscriptError("The local caption service returned invalid text or timing.") from None
        if not text:
            continue
        characters += len(text)
        if characters > MAX_TEXT_CHARS:
            raise LocalTranscriptError("The local transcript exceeds the text size limit.")
        end = round(start + duration, 3)
        start = round(start, 3)
        if end <= start:
            raise LocalTranscriptError("The local caption service returned invalid caption timing.")
        segments.append({"start": start, "end": end, "text": text})
    if not segments:
        raise LocalTranscriptError("The video returned no usable captions on this Mac.")
    return sorted(segments, key=lambda item: (item["start"], item["end"]))


def fetch_transcript(url: str, language: str, progress, *, apify_blocked: bool = False) -> dict:
    from apify_transcript import _canonical_video_url
    canonical = _canonical_video_url(url)
    if not isinstance(language, str) or not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})?", language) or language.lower() == "auto":
        raise LocalTranscriptError("Choose the video's caption language before local caption extraction.")
    video_id = parse_qs(urlsplit(canonical).query)["v"][0]
    try:
        import requests
        from urllib3.util import Timeout
        from youtube_transcript_api import (
            AgeRestricted, IpBlocked, NoTranscriptFound, PoTokenRequired,
            RequestBlocked, TranscriptsDisabled, VideoUnavailable,
            VideoUnplayable, YouTubeTranscriptApi,
        )
    except ImportError:
        raise LocalTranscriptError("Install youtube-transcript-api==1.2.4 in the app's Python environment to extract captions locally. See README.md.") from None

    class DeadlineSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.deadline = time.monotonic() + REQUEST_SECONDS
            self.max_redirects = 5
            self.trust_env = False

        def request(self, method, target, **kwargs):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise requests.Timeout("Local caption deadline reached.")
            kwargs["timeout"] = Timeout(total=remaining, connect=min(5, remaining), read=min(15, remaining))
            response = super().request(method, target, **kwargs)
            if time.monotonic() > self.deadline:
                response.close()
                raise requests.Timeout("Local caption deadline reached.")
            return response

    progress("transcript", 42, "Apify cloud blocked; extracting captions on your Mac…" if apify_blocked else "Extracting YouTube captions directly on your Mac…")
    try:
        with DeadlineSession() as session:
            api = YouTubeTranscriptApi(http_client=session)
            track = _select_track(api.list(video_id), language)
            fetched = track.fetch()
            if fetched.language_code.lower().split("-")[0] != language.lower().split("-")[0]:
                raise LocalTranscriptError("The local caption service returned a different language. No replacement language was selected.")
            segments = _normalize(fetched)
            result = {"segments": segments, "language": fetched.language_code,
                      "duration": max(item["end"] for item in segments),
                      "transcript_origin": "Local YouTube captions (Apify cloud blocked)" if apify_blocked else "Local YouTube captions (youtube-transcript-api)"}
            if apify_blocked:
                result["fallback_reason"] = "REQUEST_BLOCKED"
            return result
    except LocalTranscriptError:
        raise
    except (RequestBlocked, IpBlocked, PoTokenRequired):
        raise LocalTranscriptError("YouTube blocked caption access from this Mac. Upload the source video to transcribe it locally.") from None
    except (TranscriptsDisabled, NoTranscriptFound):
        raise LocalTranscriptError("This video has no accessible captions in the selected language. Upload the source video to transcribe it locally.") from None
    except (VideoUnavailable, VideoUnplayable, AgeRestricted):
        raise LocalTranscriptError("The video is unavailable or restricted. Use a public video with accessible captions.") from None
    except (requests.Timeout, TimeoutError):
        raise LocalTranscriptError("Local caption extraction timed out.") from None
    except requests.RequestException:
        raise LocalTranscriptError("The local caption service could not be reached.") from None
    except Exception:
        raise LocalTranscriptError("Local captions could not be extracted. Upload the source video for local transcription.") from None
