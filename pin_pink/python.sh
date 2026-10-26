#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec env -u PYTHONPATH -u LD_LIBRARY_PATH -u PYTHONHOME -u LD_PRELOAD \
    PYTHONNOUSERSITE=1 "$workspace/env/pin-pink/bin/python" "$@"
