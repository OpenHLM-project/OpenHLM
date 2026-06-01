#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_SCRIPT="$REPO_ROOT/gear_sonic_deploy/deploy.sh"

if [[ ! -f "$DEPLOY_SCRIPT" ]]; then
  echo "Could not find deploy script: $DEPLOY_SCRIPT" >&2
  exit 1
fi

cd "$REPO_ROOT"

SKIP_GIT_LFS_PULL="${SKIP_GIT_LFS_PULL:-1}" bash "$DEPLOY_SCRIPT" real --input-type zmq
