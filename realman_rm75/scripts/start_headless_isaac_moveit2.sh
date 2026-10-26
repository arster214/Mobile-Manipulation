#!/usr/bin/env bash
# One-command launcher for the native MoveIt2 RViz + cuRobo workflow:
#   - Isaac Sim backend: headless, SAFE_MODE=F by default (planning enabled)
#   - Lightweight adapter: publishes real Isaac q_actual to /joint_states
#   - Native MoveIt2 MotionPlanning plugin: Goal Ghost, markers, IK, scene
#   - Shared obstacle input: the same YAML drives Isaac, cuRobo, and MoveIt
#   - cuRobo bridge: receives only target workspace poses when Plan is clicked
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_PY="${ISAAC_PY:-/home/fang/isaacsim/python.sh}"
BACKEND_HOST="${RM75_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${RM75_BACKEND_PORT:-50075}"
BACKEND_SAFE_MODE="${RM75_BACKEND_SAFE_MODE:-F}"
PLATFORM_INITIAL_POSITION="${RM75_PLATFORM_INITIAL_POSITION:-0.5}"
LOG_DIR="${ROOT_DIR}/logs"
MOVEIT_LAUNCH="${ROOT_DIR}/moveit2/rm75_dual_moveit_config/launch/moveit_frontend.launch.py"
STARTED_BACKEND=0
mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"

python3 scripts/rm75_process_guard.py \
  --root "${ROOT_DIR}" \
  --script launch_ros2_curobo_scene.py \
  --script rm75_joint_state_adapter.py \
  --script rm75_moveit_curobo_bridge.py \
  --script rm75_curobo_collision_spheres.py \
  --script moveit_frontend.launch.py \
  --script moveit.rviz \
  --max-gpu-percent "${RM75_MAX_GPU_PERCENT:-90}" \
  --terminate

cleanup() {
  echo "[rm75] stopping launched MoveIt2 frontend processes..."
  jobs -pr | xargs -r kill 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

case "${BACKEND_SAFE_MODE}" in
  B|C|D|E|F|G) ;;
  *)
    echo "[rm75] invalid RM75_BACKEND_SAFE_MODE='${BACKEND_SAFE_MODE}', expected B..G" >&2
    exit 2
    ;;
esac

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/humble/setup.bash
  set -u
else
  echo "[rm75] warning: /opt/ros/humble/setup.bash not found; assuming ROS2 is already sourced"
fi

if pgrep -af "scripts/(rm75_rviz_interactive_markers|rm75_joint_state_adapter|rm75_moveit_curobo_bridge|rm75_curobo_collision_spheres)\.py" >/tmp/rm75_existing_frontend_processes.txt 2>/dev/null; then
  if [[ "${RM75_FORCE_START_MOVEIT_FRONTEND:-0}" != "1" ]]; then
    echo "[rm75] found an existing RM75 RViz/socket adapter; refusing to start a second /joint_states publisher." >&2
    cat /tmp/rm75_existing_frontend_processes.txt >&2
    echo "[rm75] run ./scripts/stop_rm75_stack.sh first, or set RM75_FORCE_START_MOVEIT_FRONTEND=1 if you are sure." >&2
    exit 3
  fi
fi

if python3 - <<PY
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path("${ROOT_DIR}") / "scripts"))
from rm75_obstacles import load_obstacle_scene

host = "${BACKEND_HOST}"
port = int("${BACKEND_PORT}")
try:
    with socket.create_connection((host, port), timeout=0.3) as sock:
        sock.settimeout(0.5)
        sock.sendall(b'{"command":"get_state"}\\n')
        data = b""
        while not data.endswith(b"\\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    if data:
        response = json.loads(data.decode("utf-8"))
        if response.get("success"):
            remote = (response.get("obstacle_scene") or {}).get("fingerprint")
            local = load_obstacle_scene().fingerprint
            if remote != local:
                print(
                    f"backend obstacle fingerprint {remote!r} does not match local {local!r}",
                    file=sys.stderr,
                )
                sys.exit(2)
            sys.exit(0)
    sys.exit(2)
except ConnectionRefusedError:
    sys.exit(1)
except OSError:
    sys.exit(1)
except Exception:
    sys.exit(2)
PY
then
  echo "[rm75] existing RM75 backend detected at ${BACKEND_HOST}:${BACKEND_PORT}; reusing it."
else
  probe_status=$?
  if [[ ${probe_status} -eq 2 ]]; then
    echo "[rm75] port ${BACKEND_HOST}:${BACKEND_PORT} is occupied by an incompatible RM75 backend or obstacle scene; refusing mixed scenes." >&2
    echo "[rm75] stop the complete stack and restart so Isaac, cuRobo, and MoveIt load the same config/simple_obstacles.yml." >&2
    exit 3
  fi
  if pgrep -af "launch_ros2_curobo_scene.py" >/tmp/rm75_existing_backend_processes.txt 2>/dev/null; then
    if [[ "${RM75_FORCE_START_ISAAC:-0}" != "1" ]]; then
      echo "[rm75] found existing Isaac backend process but socket is not healthy; refusing to start a second one." >&2
      cat /tmp/rm75_existing_backend_processes.txt >&2
      echo "[rm75] stop the stale process first, or set RM75_FORCE_START_ISAAC=1 if you are absolutely sure." >&2
      exit 3
    fi
  fi

  echo "[rm75] starting headless Isaac backend for MoveIt2 frontend..."
  echo "[rm75] backend log: ${LOG_DIR}/isaac_curobo_backend.log"
  RM75_SAFE_MODE="${BACKEND_SAFE_MODE}" RM75_PLATFORM_INITIAL_POSITION="${PLATFORM_INITIAL_POSITION}" \
    RM75_BACKEND_HOST="${BACKEND_HOST}" RM75_BACKEND_PORT="${BACKEND_PORT}" \
    "${ISAAC_PY}" scripts/launch_ros2_curobo_scene.py --headless \
    >"${LOG_DIR}/isaac_curobo_backend.log" 2>&1 &
  STARTED_BACKEND=1
fi

if [[ "${STARTED_BACKEND}" == "1" ]]; then
  echo "[rm75] waiting for backend ${BACKEND_HOST}:${BACKEND_PORT} ..."
  python3 - <<PY
import json
import socket
import sys
import time

host = "${BACKEND_HOST}"
port = int("${BACKEND_PORT}")
deadline = time.time() + 120.0
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=0.5) as sock:
            sock.settimeout(1.0)
            sock.sendall(b'{"command":"get_state"}\\n')
            data = b""
            while not data.endswith(b"\\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
        if data and json.loads(data.decode("utf-8")).get("success"):
            print("[rm75] backend is ready")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(0.5)
print("[rm75] backend did not become ready within 120s", file=sys.stderr)
sys.exit(1)
PY
fi

echo "[rm75] starting lightweight RM75 joint-state adapter..."
RM75_BACKEND_HOST="${BACKEND_HOST}" RM75_BACKEND_PORT="${BACKEND_PORT}" \
  python3 scripts/rm75_joint_state_adapter.py \
  >"${LOG_DIR}/moveit2_joint_state_adapter.log" 2>&1 &

echo "[rm75] waiting for the first valid /joint_states before MoveIt Goal State initialization..."
if ! timeout 15 ros2 topic echo --once /joint_states sensor_msgs/msg/JointState >/dev/null 2>&1; then
  echo "[rm75] no valid /joint_states received within 15s; refusing to initialize MoveIt from zero state." >&2
  exit 4
fi

echo "[rm75] starting pose-only MoveIt2 -> cuRobo action bridge..."
RM75_BACKEND_HOST="${BACKEND_HOST}" RM75_BACKEND_PORT="${BACKEND_PORT}" \
  python3 scripts/rm75_moveit_curobo_bridge.py --ros-args -r __ns:=/curobo \
  >"${LOG_DIR}/moveit2_curobo_bridge.log" 2>&1 &

if [[ "${RM75_SHOW_CUROBO_SPHERES:-0}" == "1" ]]; then
  echo "[rm75] starting live cuRobo collision-sphere RViz display..."
  RM75_BACKEND_HOST="${BACKEND_HOST}" RM75_BACKEND_PORT="${BACKEND_PORT}" \
    python3 scripts/rm75_curobo_collision_spheres.py \
    >"${LOG_DIR}/moveit2_curobo_collision_spheres.log" 2>&1 &
fi

echo "[rm75] launching native MoveIt2 MotionPlanning frontend..."
echo "[rm75] move_group owns scene/IK/FK; shared obstacles come from config/simple_obstacles.yml."
echo "[rm75] MoveGroupMoveAction is disabled and supplied by the cuRobo bridge."
ros2 launch "${MOVEIT_LAUNCH}" \
  allow_trajectory_execution:=false \
  use_move_group:=true \
  use_obstacles:=true \
  move_group_namespace:=curobo \
  disabled_capabilities:=move_group/MoveGroupMoveAction
