#!/usr/bin/env bash
# Render a page with headless Chromium so you can LOOK at it (read the PNG with the Read tool).
# usage: screenshot.sh "<page.html[?embed=x][#f=N]>" <out.png> [height=1400] [width=1400]
#   #f=N     jump the stepper to frame N (0-based)      ?embed=...  embed mode (no nav/title)
# Long pages: use a tall height (e.g. 8000) and crop with PIL; "#anchor" jumps give blank shots.
# Lazy-loaded iframes only render if they are inside the window, so keep the window tall for blog pages.
set -euo pipefail
target="$1"; out="$2"; h="${3:-1400}"; w="${4:-1400}"
file="${target%%[?#]*}"; rest="${target#"$file"}"
chrome="$(ls -d "$HOME"/.cache/ms-playwright/chromium-*/chrome-linux64/chrome 2>/dev/null | tail -1)"
[ -x "$chrome" ] || { echo "no Playwright Chromium found under ~/.cache/ms-playwright" >&2; exit 1; }
timeout 120 "$chrome" --headless=new --no-sandbox --disable-gpu --hide-scrollbars \
  --window-size="$w,$h" --virtual-time-budget=8000 --screenshot="$out" "file://$(realpath "$file")$rest" >/dev/null 2>&1
echo "$out"
