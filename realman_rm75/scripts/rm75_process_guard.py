#!/usr/bin/env python3
"""Clean up duplicate RM75 processes and check NVIDIA memory before startup."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time


def _proc_command(pid: int) -> str:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    return data.replace(b"\0", b" ").decode(errors="replace").strip()


def _proc_cwd(pid: int) -> str:
    try:
        return os.path.realpath(f"/proc/{pid}/cwd")
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _ancestor_pids() -> set[int]:
    ancestors = set()
    pid = os.getpid()
    while pid > 1:
        ancestors.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().split()[3])
        except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
            break
    return ancestors


def _matching_pids(script_names: list[str], root: Path) -> list[int]:
    ignored_pids = _ancestor_pids()
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in ignored_pids:
            continue
        command = _proc_command(pid)
        if not command or not any(name in command for name in script_names):
            continue
        cwd = _proc_cwd(pid)
        if cwd == str(root) or str(root) in command:
            matches.append(pid)
    return sorted(set(matches))


def _descendants(pids: list[int]) -> list[int]:
    parents = set(pids)
    found = set(pids)
    changed = True
    while changed:
        changed = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                ppid = int(entry.joinpath("stat").read_text().split()[3])
            except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
                continue
            if ppid in parents and pid not in found:
                found.add(pid)
                parents.add(pid)
                changed = True
    return sorted(found, reverse=True)


def terminate_duplicates(script_names: list[str], root: Path, grace_seconds: float = 3.0) -> int:
    matches = _matching_pids(script_names, root)
    if not matches:
        print("[rm75] no duplicate RM75 processes found")
        return 0

    targets = _descendants(matches)
    for pid in targets:
        command = _proc_command(pid)
        print(f"[rm75] stopping duplicate pid={pid}: {command}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not any(Path(f"/proc/{pid}").exists() for pid in targets):
            return len(targets)
        time.sleep(0.1)

    for pid in targets:
        if not Path(f"/proc/{pid}").exists():
            continue
        command = _proc_command(pid)
        print(f"[rm75] force stopping duplicate pid={pid}: {command}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return len(targets)


def gpu_status() -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        used, total = (int(value.strip()) for value in result.stdout.splitlines()[0].split(","))
        return used, total
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def check_gpu(max_percent: float = 90.0, wait_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + wait_seconds
    last_status = None
    while True:
        last_status = gpu_status()
        if last_status is None:
            print("[rm75] GPU check unavailable; continuing without VRAM enforcement")
            return True
        used, total = last_status
        percent = 100.0 * used / max(total, 1)
        print(f"[rm75] GPU memory: {used} MiB / {total} MiB ({percent:.1f}%)")
        if percent < max_percent or time.monotonic() >= deadline:
            return percent < max_percent
        time.sleep(0.25)


def ensure_single_instance(script_names: list[str], root: Path, max_gpu_percent: float = 90.0) -> None:
    terminate_duplicates(script_names, root)
    if not check_gpu(max_gpu_percent):
        raise RuntimeError(
            f"NVIDIA memory remains above {max_gpu_percent:.1f}% after RM75 duplicate cleanup"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", action="append", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-gpu-percent", type=float, default=90.0)
    parser.add_argument("--terminate", action="store_true")
    args = parser.parse_args()
    if args.terminate:
        terminate_duplicates(args.script, args.root)
    if not check_gpu(args.max_gpu_percent):
        print(
            f"[rm75] refusing startup: GPU memory remains above {args.max_gpu_percent:.1f}%",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
