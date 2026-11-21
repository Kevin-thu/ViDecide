#!/usr/bin/env bash
# Fetch RoboTwin 2.0 at the commit used to build RoboTwin-Plan. Only needed to render new RoboTwin-Plan episodes;
# training and evaluation use the released videos.
#
#   bash scripts/setup/setup_robotwin.sh                  # -> third_party/RoboTwin (or $ROBOTWIN_ROOT)
#
# RoboTwin needs its own Python environment (Python 3.10; we used SAPIEN 3.0.0b1, mplib 0.2.1, torch 2.4.1):
#   conda create -n robotwin python=3.10 -y && conda activate robotwin
#   cd third_party/RoboTwin && bash scripts/_install.sh && bash scripts/_download_assets.sh
# The released episodes were planned with RoboTwin's default planner of the Aloha-AgileX robot (cuRobo, installed by
# _install.sh). Without cuRobo, pass `-- --planner mplib_RRT` to render_episodes.py (collect_videos.py then stubs the
# cuRobo imports and plans with mplib).
# Then render from the ViDecide root with that environment's python, e.g.
#   ROBOTWIN_PYTHON=$(conda run -n robotwin which python) python scripts/robotwin/render_episodes.py --help
set -euo pipefail
cd "$(dirname "$0")/../.."

RT_URL="${RT_URL:-https://github.com/RoboTwin-Platform/RoboTwin.git}"
RT_COMMIT=ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
DEST="${ROBOTWIN_ROOT:-third_party/RoboTwin}"

if [[ ! -d "$DEST/.git" ]]; then
  git clone --quiet "$RT_URL" "$DEST"
fi
if ! git -C "$DEST" cat-file -e "$RT_COMMIT^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch --quiet origin
fi
git -C "$DEST" checkout --quiet --detach "$RT_COMMIT"
echo "RoboTwin $(git -C "$DEST" rev-parse --short HEAD) in $DEST"
echo "Next: create its environment and download its assets (see the comments at the top of this script)."
