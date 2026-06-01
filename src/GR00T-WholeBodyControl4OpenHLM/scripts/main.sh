      
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

if [[ -n "${TMUX:-}" ]]; then
  session="$(tmux display-message -p '#S')"
  tmux new-window -t "$session" -n "twist2" "bash -ilc '$cmd1'"
  tmux split-window -h -t "$session:twist2" "bash -ilc '$cmd2'"
  tmux split-window -v -t "$session:twist2.0" "bash -ilc '$cmd3'"
  tmux split-window -v -t "$session:twist2.1" "bash -ilc '$cmd4'"
  tmux select-layout -t "$session:twist2" tiled
else
  session="twist2_$$"
  tmux new-session -d -s "$session" "bash -ilc '$cmd1'"
  tmux split-window -h -t "$session" "bash -ilc '$cmd2'"
  tmux split-window -v -t "$session:0.0" "bash -ilc '$cmd3'"
  tmux split-window -v -t "$session:0.1" "bash -ilc '$cmd4'"
  tmux select-layout -t "$session" tiled
  tmux attach -t "$session"
fi

    