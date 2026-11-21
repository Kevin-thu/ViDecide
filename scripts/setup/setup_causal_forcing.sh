#!/usr/bin/env bash
# Fetch Causal-Forcing at the commit ViDecide is built on and overlay our modified and added files.
#
#   bash scripts/setup/setup_causal_forcing.sh            # -> third_party/Causal-Forcing
#   CAUSAL_FORCING_ROOT=/path/to/checkout bash scripts/setup/setup_causal_forcing.sh
#
# Our files are in third_party/causal_forcing_overlay (see NOTICE for what changed). The scripts find the patched
# checkout in third_party/Causal-Forcing, or wherever CAUSAL_FORCING_ROOT points.
set -euo pipefail
cd "$(dirname "$0")/../.."

CF_URL="${CF_URL:-https://github.com/thu-ml/Causal-Forcing.git}"
CF_COMMIT=3e7cc998e745afd439d5b691e30b8a4a5d48ce38
DEST="${CAUSAL_FORCING_ROOT:-third_party/Causal-Forcing}"
OVERLAY=third_party/causal_forcing_overlay

if [[ ! -d "$DEST/.git" ]]; then
  if [[ -e "$DEST" ]]; then
    echo "$DEST exists but is not a git checkout; remove it or set CAUSAL_FORCING_ROOT" >&2; exit 1
  fi
  git clone --quiet "$CF_URL" "$DEST"
fi
if ! git -C "$DEST" cat-file -e "$CF_COMMIT^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch --quiet origin
fi

# Only files of the overlay may differ from the pinned commit (so that re-running the script is safe).
overlay_files=$(cd "$OVERLAY" && find . -type f -not -path "*/__pycache__/*" | sed 's|^\./||' | sort)
changed=$(git -C "$DEST" status --porcelain --untracked-files=no | awk '{print $2}' | sort)
foreign=$(comm -23 <(echo "$changed" | grep . || true) <(echo "$overlay_files"))
if [[ -n "$foreign" ]]; then
  echo "local changes outside the overlay in $DEST:" >&2; echo "$foreign" >&2
  echo "commit or discard them first" >&2; exit 1
fi
git -C "$DEST" checkout --quiet -- . 2>/dev/null || true
git -C "$DEST" checkout --quiet --detach "$CF_COMMIT"

while read -r f; do
  mkdir -p "$DEST/$(dirname "$f")"; cp "$OVERLAY/$f" "$DEST/$f"
done <<< "$overlay_files"
n=0
while read -r f; do
  cmp -s "$OVERLAY/$f" "$DEST/$f" || { echo "overlay copy failed for $f" >&2; exit 1; }
  n=$((n + 1))
done <<< "$overlay_files"
echo "Causal-Forcing $(git -C "$DEST" rev-parse --short HEAD) in $DEST with $n ViDecide files"
