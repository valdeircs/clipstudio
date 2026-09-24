#!/bin/zsh
set -e
cd "${0:A:h}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --upgrade 'yt-dlp[default]'
print '\nYouTube support updated. Restart ClipStudio if it is running.'
read "?Press Return to close. "
