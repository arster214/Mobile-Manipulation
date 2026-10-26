#!/usr/bin/env bash
# Snapshot ROS2 graph health for the RViz2 + Isaac/cuRobo workflow.
set -euo pipefail

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

section() {
  printf '\n========== %s ==========' "$1"
  printf '\n'
}

section "nodes"
ros2 node list || true

section "/joint_states publishers"
ros2 topic info /joint_states -v || true

section "/tf publishers"
ros2 topic info /tf -v || true

section "/tf_static publishers"
ros2 topic info /tf_static -v || true

section "state/tf related nodes"
ros2 node list | grep -E 'robot_state|joint_state|tf|rm75' || true

section "topic hz: /joint_states (5s)"
timeout 5s ros2 topic hz /joint_states || true

section "topic bw: /joint_states (5s)"
timeout 5s ros2 topic bw /joint_states || true

section "topic hz: /tf (5s)"
timeout 5s ros2 topic hz /tf || true

section "topic bw: /tf (5s)"
timeout 5s ros2 topic bw /tf || true

section "rm75 topics"
ros2 topic list | grep -E '^/rm75|^/joint_states|^/tf' || true

section "rm75 services"
ros2 service list | grep -E '^/rm75' || true
