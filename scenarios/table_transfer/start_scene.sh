#!/usr/bin/env bash
set -euo pipefail
SCENARIO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/fang/isaacsim/python.sh "$SCENARIO_DIR/scripts/launch_scene.py" "$@"
