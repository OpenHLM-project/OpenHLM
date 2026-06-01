#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_ACTIVATE="$REPO_ROOT/.venv_teleop/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Could not find teleop venv: $VENV_ACTIVATE" >&2
  echo "Run from repo root: bash install_scripts/install_pico.sh" >&2
  exit 1
fi

source "$VENV_ACTIVATE"
cd "$REPO_ROOT"

python gear_sonic/scripts/pico_thread.py --visualize --visualize_human_motion