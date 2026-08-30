"""Keeps run_program.py alive.

The orchestrator makes network calls into the Azure SDK, and some of those have
no timeout. One wedged for 48 minutes with three free compute slots and no error
in the log, which is silent damage: the queue simply stops draining.

Liveness is judged by the mtime of program_state.json, which the orchestrator
rewrites every poll cycle (~60s). The log is NOT a valid signal -- it only gets
written on events, and a legitimate quiet stretch while three models train can
run for hours.
"""
from __future__ import annotations

import datetime
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STATE = RESULTS / "program_state.json"
PY = ROOT / ".venv" / "bin" / "python"

CHECK_SECONDS = 120
STALE_SECONDS = 600  # ten missed poll cycles


def note(msg: str) -> None:
    line = f"{datetime.datetime.now():%H:%M:%S} [watchdog] {msg}"
    print(line, flush=True)
    with open(RESULTS / "program.log", "a") as fh:
        fh.write(line + "\n")


def spawn() -> subprocess.Popen:
    out = open(RESULTS / "program.stdout", "a")
    return subprocess.Popen(
        [str(PY), str(ROOT / "tools" / "run_program.py")],
        stdout=out, stderr=subprocess.STDOUT, cwd=str(ROOT))


def restart(proc: subprocess.Popen | None) -> subprocess.Popen:
    if proc and proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            note("orchestrator ignored kill")
    proc = spawn()
    note(f"orchestrator running as pid {proc.pid}")
    return proc


def main() -> None:
    proc = restart(None)
    while True:
        time.sleep(CHECK_SECONDS)

        if proc.poll() is not None:
            if proc.returncode == 0:
                note("orchestrator exited cleanly; queue drained, watchdog done")
                return
            note(f"orchestrator died rc={proc.returncode}")
            proc = restart(proc)
            continue

        if STATE.exists():
            stale = time.time() - STATE.stat().st_mtime
            if stale > STALE_SECONDS:
                note(f"state untouched for {stale / 60:.0f}m; assuming wedged")
                proc = restart(proc)


if __name__ == "__main__":
    main()
