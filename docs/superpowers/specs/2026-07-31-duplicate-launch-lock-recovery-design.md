# Duplicate-Launch Lock Recovery Design

## Evidence and problem

An existing `scripts/run_gui.py` process owns
`D:\LTX_Supervisor_Storage\state\supervisor.lock`. On Windows, the current
acquisition path reads byte zero before its protected `msvcrt.locking` call.
Reading a byte locked by another process raises `PermissionError`, so the normal
single-controller collision escapes as a traceback instead of the intended
`OwnershipError`.

The running process is the fail-closed acceptance-review application: its
acceptance endpoints answer, `/api/status` returns 404, and ComfyUI has no
running or pending prompts.

## Selected behavior

`SupervisorInstanceLock.acquire()` will determine whether the sentinel file is
empty without reading the locked byte. The nonblocking byte lock remains the
sole ownership decision. Any Windows lock denial is translated to the existing
`OwnershipError`; the lock is never bypassed or deleted.

`scripts/run_gui.py` will handle only an initial instance-lock collision. It
will log that another GUI owns the pipeline, open the existing GUI URL unless
`--no-browser` was supplied, and return success. Ownership errors raised later
for legacy-process takeover, busy queues, or stale node contracts retain their
current failure behavior.

## Safety and verification

- Never start a second supervisor or forcibly steal a lock.
- Never delete the persistent lock file; OS lock ownership, not file presence,
  determines whether a controller is live.
- Add a Windows regression where reading the locked byte would raise
  `PermissionError`, while the actual lock denial must become `OwnershipError`.
- Add launcher regressions for clean duplicate exit, browser opening, and
  `--no-browser` behavior.
- After tests pass, stop only the verified review-only PID, confirm the ComfyUI
  queue is still empty, and start the updated project GUI. No ComfyUI restart is
  required.
