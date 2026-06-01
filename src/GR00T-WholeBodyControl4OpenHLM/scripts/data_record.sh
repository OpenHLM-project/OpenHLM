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
# task_name="0819_shelf"

robot_ip="192.168.123.164"
data_frequency=30
task_name="20260521_1102_test"
task_desc="test"

python gear_sonic/scripts/run_openhlm_data_record.py \
    --frequency "${data_frequency}" \
    --robot_ip "${robot_ip}" \
    --task_name "${task_name}" \
    --desc "${task_desc}" \
# --rerun_visualize 
