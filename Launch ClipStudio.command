#!/bin/zsh
set -e
cd "${0:A:h}"
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"

# Keep a custom FFmpeg already on PATH; upgrade the default Homebrew selection.
CLIPSTUDIO_FFMPEG="$(command -v ffmpeg || true)"
if [[ -z "$CLIPSTUDIO_FFMPEG" || "$CLIPSTUDIO_FFMPEG" == "/opt/homebrew/bin/ffmpeg" || "$CLIPSTUDIO_FFMPEG" == "/usr/local/bin/ffmpeg" ]]; then
  for CLIPSTUDIO_FFMPEG_DIR in /opt/homebrew/opt/ffmpeg-full/bin /usr/local/opt/ffmpeg-full/bin; do
    if [[ -x "$CLIPSTUDIO_FFMPEG_DIR/ffmpeg" && -x "$CLIPSTUDIO_FFMPEG_DIR/ffprobe" ]]; then
      export PATH="$CLIPSTUDIO_FFMPEG_DIR:$PATH"
      break
    fi
  done
fi

if [[ -x ".venv/bin/python" ]]; then
  CLIPSTUDIO_PYTHON=".venv/bin/python"
  export PATH="$PWD/.venv/bin:$PATH"
elif [[ -n "$VIRTUAL_ENV" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  CLIPSTUDIO_PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x "/opt/homebrew/bin/python3" ]]; then
  CLIPSTUDIO_PYTHON="/opt/homebrew/bin/python3"
elif [[ -x "/usr/local/bin/python3" ]]; then
  CLIPSTUDIO_PYTHON="/usr/local/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  CLIPSTUDIO_PYTHON="$(command -v python3)"
else
  print "ClipStudio needs Python 3.10 or newer. Follow README.md, then open this launcher again."
  read "?Press Return to close. "
  exit 1
fi

exec "$CLIPSTUDIO_PYTHON" server.py --open-browser
