#!/usr/bin/env bash
# Stop RM75 launcher/backend/frontend processes from this workspace.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher_patterns=(
  "scripts/start_headless_isaac_moveit2.sh"
  "scripts/start_headless_isaac_rviz.sh"
  "scripts/start_rviz_frontend.sh"
)
patterns=(
  "${launcher_patterns[@]}"
  "scripts/launch_ros2_curobo_scene.py"
  "scripts/rm75_rviz_interactive_markers.py"
  "scripts/rm75_joint_state_adapter.py"
  "scripts/rm75_moveit_curobo_bridge.py"
  "scripts/rm75_moveit_obstacles.py"
  "moveit2/rm75_dual_moveit_config/launch/moveit_frontend.launch.py"
  "move_group.*__ns:=/curobo"
  "rviz2.*${ROOT_DIR}/moveit2/"
)

signal_matches() {
  local signal="$1"
  local label="$2"
  shift 2
  local pattern pid cmd
  for pattern in "$@"; do
    while read -r pid cmd; do
      [[ -z "${pid:-}" ]] && continue
      found=1
      echo "[rm75] ${label} pid=${pid}: ${cmd}"
      kill "-${signal}" "${pid}" 2>/dev/null || true
    done < <(pgrep -af "${pattern}" | awk -v self="$$" '$1 != self {pid=$1; $1=""; sub(/^ /, ""); print pid, $0}')
  done
}

found=0
signal_matches TERM "stopping launcher" "${launcher_patterns[@]}"
if [[ "${found}" == "1" ]]; then
  sleep 5
fi
signal_matches TERM "stopping remaining process" "${patterns[@]}"

if [[ "${found}" == "0" ]]; then
  echo "[rm75] no RM75 stack processes found"
else
  sleep 5
  signal_matches KILL "force stopping process" "${patterns[@]}"
fi

LOCK_PATH="${RM75_BACKEND_LOCK:-/tmp/rm75_isaac_backend.lock}"
if [[ -f "${LOCK_PATH}" ]] && ! pgrep -af "scripts/launch_ros2_curobo_scene.py" >/dev/null; then
  echo "[rm75] removing stale lock ${LOCK_PATH}"
  rm -f "${LOCK_PATH}"
fi
