#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3 scripts/rm75_process_guard.py \
  --root "${ROOT_DIR}" \
  --script rm75_rviz_interactive_markers.py \
  --script rviz/rm75_interactive_markers.rviz \
  --max-gpu-percent "${RM75_MAX_GPU_PERCENT:-90}" \
  --terminate

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

URDF_PATH="$(python3 scripts/prepare_rviz_urdf.py)"

cleanup() {
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ros2 run robot_state_publisher robot_state_publisher "$URDF_PATH" &
python3 scripts/rm75_rviz_interactive_markers.py &
rviz2 -d rviz/rm75_interactive_markers.rviz
