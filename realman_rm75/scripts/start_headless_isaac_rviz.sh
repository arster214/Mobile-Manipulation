#!/usr/bin/env bash
# One-command launcher:
#   - Isaac Sim + cuRobo backend: headless
#   - ROS2 Humble robot_state_publisher + RViz2 + Interactive Markers: headed
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_PY="${ISAAC_PY:-/home/fang/isaacsim/python.sh}"
BACKEND_HOST="${RM75_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${RM75_BACKEND_PORT:-50075}"
SAFE_MODE="${RM75_SAFE_MODE:-G}"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
STARTED_BACKEND=0

cd "${ROOT_DIR}"

python3 scripts/rm75_process_guard.py \
  --root "${ROOT_DIR}" \
  --script launch_ros2_curobo_scene.py \
  --script rm75_rviz_interactive_markers.py \
  --script rviz/rm75_interactive_markers.rviz \
  --max-gpu-percent "${RM75_MAX_GPU_PERCENT:-90}" \
  --terminate

cleanup() {
  echo "[rm75] stopping launched processes..."
  jobs -pr | xargs -r kill 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

case "${SAFE_MODE}" in
  A|B|C|D|E|F|G) ;;
  *)
    echo "[rm75] invalid RM75_SAFE_MODE='${SAFE_MODE}', expected A..G" >&2
    exit 2
    ;;
esac

echo "[rm75] SAFE_MODE=${SAFE_MODE}"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/humble/setup.bash
  set -u
else
  echo "[rm75] warning: /opt/ros/humble/setup.bash not found; assuming ROS2 is already sourced"
fi

if [[ "${SAFE_MODE}" != "A" ]]; then
  if python3 - <<PY
import json
import socket
import sys

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
    if data and json.loads(data.decode("utf-8")).get("success"):
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
    echo "[rm75] existing RM75 backend detected at ${BACKEND_HOST}:${BACKEND_PORT}; reusing it instead of starting another Isaac."
  else
    probe_status=$?
    if [[ ${probe_status} -eq 2 ]]; then
      echo "[rm75] port ${BACKEND_HOST}:${BACKEND_PORT} is occupied but does not look like an RM75 backend; refusing to start Isaac." >&2
      echo "[rm75] free the port or set RM75_BACKEND_PORT to another value." >&2
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

    echo "[rm75] starting headless Isaac backend..."
    echo "[rm75] backend log: ${LOG_DIR}/isaac_curobo_backend.log"
    RM75_SAFE_MODE="${SAFE_MODE}" RM75_BACKEND_HOST="${BACKEND_HOST}" RM75_BACKEND_PORT="${BACKEND_PORT}" \
      "${ISAAC_PY}" scripts/launch_ros2_curobo_scene.py --headless \
      >"${LOG_DIR}/isaac_curobo_backend.log" 2>&1 &
    ISAAC_PID=$!
    STARTED_BACKEND=1
  fi

  # Wait until the backend socket is accepting commands. This keeps RViz from
  # coming up before Isaac has loaded the robot and cuRobo adapter.
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
            sock.sendall(b'{"command":"get_state"}\n')
            data = b""
            while not data.endswith(b"\n"):
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
else
  echo "[rm75] SAFE_MODE=A: not starting Isaac backend"
fi

URDF_PATH="$(python3 scripts/prepare_rviz_urdf.py)"
echo "[rm75] RViz URDF: ${URDF_PATH}"

echo "[rm75] starting robot_state_publisher..."
ros2 run robot_state_publisher robot_state_publisher "${URDF_PATH}" \
  >"${LOG_DIR}/robot_state_publisher.log" 2>&1 &

if [[ "${SAFE_MODE}" == "A" ]]; then
  if ros2 pkg executables joint_state_publisher >/dev/null 2>&1; then
    echo "[rm75] SAFE_MODE=A: starting static joint_state_publisher for zero-position RobotModel..."
    ros2 run joint_state_publisher joint_state_publisher \
      >"${LOG_DIR}/joint_state_publisher_static.log" 2>&1 &
  else
    echo "[rm75] SAFE_MODE=A: joint_state_publisher not found; RobotModel may show only fixed links"
  fi
else
  echo "[rm75] starting RViz frontend adapter..."
  RM75_SAFE_MODE="${SAFE_MODE}" RM75_BACKEND_HOST="${BACKEND_HOST}" RM75_BACKEND_PORT="${BACKEND_PORT}" \
    python3 scripts/rm75_rviz_interactive_markers.py \
    >"${LOG_DIR}/rviz_interactive_markers.log" 2>&1 &
fi

echo "[rm75] launching RViz2..."
echo "[rm75] SAFE_MODE ladder: A=static, B=joint states, C=markers, D=IK preview, F=plan, G=execute."
echo "[rm75] In mode C+: use Interact, drag blue/orange balls. In D+ /rm75/ik_preview shows a visual ghost. In F/G right-click for Plan/Execute."
rviz2 -d rviz/rm75_interactive_markers.rviz
