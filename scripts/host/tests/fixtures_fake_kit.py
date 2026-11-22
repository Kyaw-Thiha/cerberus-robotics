"""Fake Kit launcher for kit_retry tests: fails N times, or hangs, then reports Startup Complete."""
import os
import pathlib
import sys
import time

state = pathlib.Path(os.environ["FAKE_STATE"])
n = int(state.read_text()) if state.exists() else 0
state.write_text(str(n + 1))
log = pathlib.Path(os.environ["CERBERUS_KIT_LOG"])
log.parent.mkdir(parents=True, exist_ok=True)
mode = os.environ.get("FAKE_MODE", "fail")
with log.open("a") as fh:
    if mode == "hang":
        fh.write("2026-01-01T00:00:00Z [1ms] [Verbose] loading extensions\n")
        fh.flush()
        time.sleep(60)
        sys.exit(0)
    if mode == "fail" and n < int(os.environ.get("FAKE_FAIL_UNTIL", "0")):
        fh.write("2026-01-01T00:00:01Z [1ms] [Fatal] libcarb.tasking make_fcontext "
                 "terminating this process with exit code 139.\n")
        sys.exit(139)
    fh.write("2026-01-01T00:00:02Z [2ms] [Info] Simulation App Startup Complete\n")
    time.sleep(0.1)
sys.exit(0)
