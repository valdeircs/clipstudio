# Source and dependency notes

This is a private application repository. No open-source license is granted for the original ClipStudio application by this repository. The owner can choose a distribution license later.

ClipStudio calls separately installed tools and libraries; their source code, package trees, model weights and executable binaries are not bundled here. Their upstream licenses continue to apply:

| Component | License / upstream information |
| --- | --- |
| Pillow | MIT-CMU — https://github.com/python-pillow/Pillow |
| youtube-transcript-api | MIT — https://github.com/jdepoix/youtube-transcript-api |
| yt-dlp | Unlicense for its core; see upstream notices for dependencies — https://github.com/yt-dlp/yt-dlp |
| OpenAI Whisper, optional | MIT — https://github.com/openai/whisper |
| FFmpeg, installed separately | LGPL/GPL depending on the installed build — https://ffmpeg.org/legal.html |
| @napi-rs/keyring, optional keychain connector | MIT — package metadata and notices apply |
| OpenCV, optional preinstalled detector | See the installed version's license; not distributed here |
| macOS Vision | Apple platform framework; installed with macOS, not distributed here |

The built-in demo is the original “Make every second count” app demonstration created for this project. It is separate from user-imported YouTube videos. User source footage, transcripts, private connection settings, exports, thumbnails and logs are excluded.

The repository does not grant rights to third-party video footage. Users supply or access their own source media.
