#!/usr/bin/env python3
"""Generate an RViz-friendly URDF with absolute file:// mesh URIs."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "urdf" / "rm75_dual_rmg24_sim.urdf"
OUTPUT = ROOT / "build" / "rm75_rviz.urdf"


def main() -> None:
    text = SOURCE.read_text()

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        if raw.startswith(("package://", "file://", "/")):
            path = raw
        else:
            path = (SOURCE.parent / raw).resolve().as_posix()
        if path.startswith("/"):
            path = "file://" + path
        return f'filename="{path}"'

    text = re.sub(r'filename="([^"]+)"', repl, text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
    print(OUTPUT)


if __name__ == "__main__":
    main()
