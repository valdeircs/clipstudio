# ClipStudio

Turn YouTube videos into vertical clips, Portuguese or other source-language captions, subtitle files, and thumbnails.

## Open

Double-click **Launch ClipStudio.command** and keep its Terminal window open. The app runs at **http://127.0.0.1:8765**. Use a regular browser for video playback; Codex's embedded browser crashed during a playback test.

## Default: free transcript extraction, then Gemini

ClipStudio uses the free open-source [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) library directly on your Mac. Getting existing captions requires no Apify run, transcript API key, or paid scraping service.

1. In **Connections**, save your Gemini key for the separate AI selection step.
2. Keep **Free transcript → Gemini** as the workflow, then paste a YouTube link.
3. ClipStudio fetches the timestamped captions locally. Gemini makes one request to choose complete moments, independently setting each duration between 30 and 90 seconds. It returns titles and explains the opening and ending of each selection.
4. Review the clips, then render. The full video downloads only when needed for export; captions and thumbnails are created locally.

Gemini usage follows your Google account's limits. Caption extraction itself is free. A video must have accessible captions in the chosen language. For videos without captions, **Free local rules · no AI** downloads the video and uses Whisper speech transcription when needed; that mode also works without a Gemini key.

## Optional: Apify transcript workflow

1. Open **Connections**. ClipStudio can reuse your existing Apify CLI login from the macOS Keychain; you can also enter an Apify token in the local form or set `APIFY_TOKEN`.
2. Enter your **Gemini API key in the app**, not in chat. You can instead set `GEMINI_API_KEY` or `GOOGLE_API_KEY` before starting the app.
3. The default uses your own Apify actor. If a third-party actor is selected, explicitly allow **up to $0.02 of Apify usage per video** in Connections. Saving settings does not start a run.
4. Paste the YouTube link, choose language and suggestion count, then select **Apify transcript → Gemini**.
5. Apify fetches the timestamped transcript. Gemini makes one analysis request to select complete moments and return clip titles, reasons and transcript boundaries.
6. Review the suggestions. The video downloads only when you render the first clip. Matching downloaded videos are reused from the local workspace.
7. Adjust framing and export MP4, PNG cover and SRT subtitles.

The default actor is your own `agency-shift/youtube-transcript-scraper`; its timestamped transcript output is supported. The third-party usage consent gate applies only when a different actor is selected. The default AI model is `gemini-3.5-flash-lite`. Model selection is editable in Connections. The AI receives the transcript, not the full video; it therefore evaluates spoken content, not visual expressions or camera changes.

Apify and Gemini are external services. Their quotas and charges depend on your accounts. **The optional Apify workflow is not guaranteed to be free.** Free-tier access, where available, is subject to model and account limits; ClipStudio does not verify your account’s entitlement. Your own actor has no third-party actor fee, but Apify compute and storage still consume account usage, using included credits or billing. ClipStudio retains a $0.02 cap for each Apify actor run and a bounded runtime; it will not silently retry an uncertain charge or fall back to another AI provider. AI requests have no paid rendering component, but the app cannot determine your Gemini account's billing tier.

If the owned actor returns `REQUEST_BLOCKED` because YouTube blocks its cloud IP, ClipStudio tries transcript-only extraction on your Mac using `youtube-transcript-api==1.2.4`, then makes the same single Gemini analysis request. This does not download the full video, retry Apify, or use another paid service; video downloading still waits until export. Other errors, missing captions, and failures from custom actors fail normally without this fallback.

Existing Apify run IDs and transcript caches can be reused on retry. Invalid AI results fail explicitly; they are not presented as successful analysis. Clip suggestions require editorial review and are not predictions of virality.

## More accurate selection and caption styles

AI selection has no fixed duration target: every suggested moment must be 30–90 seconds, begin with enough context, and finish its idea or story. Word timestamps are used to split caption cues into sentence units where the timing is reliable; plain captions keep their original cue boundaries. The app rejects out-of-range clips and known unfinished boundaries. It removes overlapping duplicates by AI score while preserving the strongest distinct selections. Transcript timestamps can still differ from the actual speech, so review the cut before publishing.

Choose a **Message goal** without writing a prompt: Best complete moments, Inspiring, Reflective, Encouraging, Teaching, or Testimony. **Content context** can be automatically inferred or set to Church / faith. Church selections preserve the speaker's spiritual context and complete message; the selected goal is shown with the results.

The AI selects from prevalidated sentence ranges instead of calculating its own timestamps. Each allowed range is already 30–90 seconds. The selection prompt asks Gemini to read the following context and retain reversals or qualifications. Exact opening and closing quotes are checked against the transcript. If an advisory range ID disagrees, the app only corrects it when both quotes identify one unique legal range, and records the correction. This validates source grounding and timing; editorial context still needs review.

For an existing project, use **Reselect with Gemini** to analyze its saved transcript with one request. This also upgrades earlier rule-based suggestions. No new transcript scrape or video download is needed. If analysis fails, the current clips stay available; successful revisions preserve earlier clip metadata and export files. API usage follows your selected account's pricing.

Framing supports **Automatic**, **Wide (1×)**, **Medium (1.25×)**, **Close (1.5×)**, and **Manual (1–2×)** zoom. Manual controls adjust horizontal and vertical placement. On macOS, automatic framing uses the built-in Vision face detector when Swift is available (or an installed OpenCV detector). Analysis stays on the Mac. Automatic framing samples the clip to choose one conservative crop; it does not track a moving speaker throughout the shot. If local face detection is unavailable or unsuitable, the renderer explains its fallback. Keep full video / backdrop preserves the whole source frame and disables zoom.

Six burned-caption styles are available: **Bold yellow**, **Clean white**, **White cards**, **Strong outline**, **Neon mint**, and **Minimal**. Each supports small/medium/large type and lower/middle/top placement. The editor shows an approximate style preview; the rendered video contains the timed captions. Change a style, then render again to update the MP4. SRT remains plain subtitle text.

## Fully local alternative

Choose **Free local rules · no AI** to use installed yt-dlp, cached Whisper and transparent transcript scoring without external AI API calls. Demo videos and uploads use this workflow. Local analysis also supports source-language captions, framing, thumbnails and the same MP4/SRT exports.

This mode has no subscription or per-export fees. Your Mac supplies CPU, storage and electricity. Internet is required for YouTube downloads and initial software/model installation.

## Setup on another Mac

On this Mac, the required video tools and speech models were already installed. For another Mac with Homebrew:

```sh
brew install python ffmpeg-full deno node
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
# Optional fallback speech transcription:
brew install openai-whisper
npm --prefix integrations ci --omit=dev --ignore-scripts
```

The speech model is needed only for the local workflow when usable captions are unavailable. The optional Node keychain integration reads the existing Apify CLI credential, keeping it in memory; a manually supplied Apify token works without it. The launcher selects this app's Python environment first. When installed, the Homebrew `ffmpeg-full` build is preferred over the standard Homebrew FFmpeg link; an explicit custom FFmpeg on PATH is preserved.

If YouTube downloads fail, double-click **Update YouTube.command**, restart the app, and retry. This updates the app's own downloader without changing system tools. YouTube may still restrict some sources; upload a video file you can access as the local alternative.

## Local processing and GitHub

The application and FFmpeg worker run on this Mac. The private GitHub repository contains application source and the built-in demo; your videos, transcripts, exports, keys and runtime environments stay local. Lovable hosting is not configured. A future hosted interface would need a reachable, authenticated video worker.

## Files and privacy

- Source videos and exports are saved in `data/projects` beside the app; uploads use `data/uploads`.
- Manually entered keys are stored in `data/connections.json` with restricted permissions. They are never returned by the API, stored in project metadata, or bundled in the downloadable app ZIP.
- External requests send only the YouTube URL/transcript needed for the selected workflow. Local video rendering does not upload video to a processing service.
- The server binds to loopback and checks Origin, Host and mutation headers. Do not expose it publicly.
- Diagnostics remain local in project folders and are not downloadable through media routes.
- Keep the app open during processing. Sources are limited to two hours/2 GB; AI selections are 30–90 seconds; manual trimming and older local clips support 1–90 seconds. Jobs run sequentially.
- Use videos you own or have permission to edit. Publishing to TikTok or Instagram is manual.

## Verification

Run the portable, offline regression suite from the app directory with `.venv/bin/python -m unittest discover -s tests`.

The bundled selection and re-selection suite passes 31 focused tests, including incorrect range IDs, unique quote recovery, ambiguous quotes, 30–90-second limits, and preserving existing clips after failed analysis. Six caption styles and automatic/manual framing have local regression checks. A live Gemini test using the first 160 seconds of a Portuguese sermon selected an 80.621-second complete passage; that cached selection was exported by ClipStudio without another API request. The final opening was tightened by 0.25 seconds to the first selected word, producing an approximately 80.37-second test export at 1080 × 1920 with H.264 video, AAC audio, captions, SRT and a portrait cover. This verifies a short excerpt, not the full 30-minute source or the quality of every future AI selection. Source captions can contain transcription mistakes.


A real YouTube import and Portuguese transcript retrieval succeeded after updating yt-dlp to 2026.8.19. Local sample exports were checked as valid H.264/AAC MP4 with burned captions, PNG covers and SRT subtitles. API adapters and backend flows have mocked tests for credentials, validation, charge limits, structured AI responses, transcript caching/resume and downloading only at export. The owned actor was deployed and built successfully; YouTube blocked the tested cloud and existing datacenter-proxy connections. Both direct local extraction and the cloud-block fallback were verified live with all 278 Portuguese caption segments and cache reuse. The default direct workflow made zero Apify requests and downloaded no video during analysis. Gemini was mocked for that integration check; it does not verify a live Gemini response or account quota. The optional own-actor workflow uses the connected Apify account without the third-party consent checkbox; selecting any other actor requires explicit usage consent.

References: [Your private Apify actor](https://console.apify.com/actors/l7L3SKuxxoP9l35Hb), [Apify run API](https://docs.apify.com/api/v2/actors-runs-post), [Gemini models](https://ai.google.dev/gemini-api/docs/models), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [Whisper](https://github.com/openai/whisper), [FFmpeg](https://ffmpeg.org/legal.html).
