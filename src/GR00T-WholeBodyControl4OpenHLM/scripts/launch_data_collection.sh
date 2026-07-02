#!/usr/bin/env bash
set -euo pipefail

file_dir="$(dirname "$(readlink -f "$BASH_SOURCE")")"
init_cmd='source ~/.bashrc; sleep 1'
base_cmd="$init_cmd; cd $file_dir"
cmd1="$base_cmd; bash deploy_stream.sh; exec bash"
cmd2="$base_cmd; bash pico_stream_pure.sh; exec bash"
cmd3="$base_cmd; bash data_record.sh; exec bash"
cmd4="$base_cmd; bash run_camera_teleop.sh; exec bash"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Please install tmux and retry." >&2
  exit 1
fi

create_panes() {
  local deploy_pane="$1"
  local pico_pane

  pico_pane="$(tmux split-window -h -P -F '#{pane_id}' -t "$deploy_pane" "bash -ilc '$cmd2'")"
  tmux split-window -v -P -F '#{pane_id}' -t "$deploy_pane" "bash -ilc '$cmd3'" >/dev/null
  tmux split-window -v -P -F '#{pane_id}' -t "$pico_pane" "bash -ilc '$cmd4'" >/dev/null
  tmux select-pane -t "$deploy_pane"
}

if [[ -n "${TMUX:-}" ]]; then
  session="$(tmux display-message -p '#S')"
  deploy_pane="$(tmux new-window -P -F '#{pane_id}' -t "$session" -n "OpenHLM" "bash -ilc '$cmd1'")"
  create_panes "$deploy_pane"
else
  session="OpenHLM"
  deploy_pane="$(tmux new-session -d -P -F '#{pane_id}' -s "$session" "bash -ilc '$cmd1'")"
  create_panes "$deploy_pane"
  tmux attach -t "$session"
fi
