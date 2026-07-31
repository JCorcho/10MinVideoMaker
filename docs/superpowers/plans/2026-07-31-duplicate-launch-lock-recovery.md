# Duplicate-Launch Lock Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a duplicate Windows one-click launch open the existing GUI and exit cleanly without weakening single-controller ownership.

**Architecture:** Keep `msvcrt.locking` as the ownership authority, remove the pre-lock byte read, and catch only the initial GUI lock collision. Existing later ownership failures remain fatal.

**Tech Stack:** Python 3.12, Windows `msvcrt`, `unittest`, FastAPI launcher.

## Global Constraints

- Work only in the authorized repository and project-owned D-drive storage.
- Do not delete or bypass the lock file.
- Do not interrupt a render, restart ComfyUI, or start a second controller.
- Use test-driven development and `apply_patch` for source edits.

---

### Task 1: Normalize Windows lock collision

**Files:**
- Create: `tests/test_ownership.py`
- Modify: `tenminvideomaker/ownership.py`

**Interfaces:**
- Consumes: `SupervisorInstanceLock.acquire()`.
- Produces: `OwnershipError("Another 10MinVideoMaker controller is already running.")` for a byte locked by another process.

- [ ] Write a failing test whose fake handle raises `PermissionError` on `read(1)` and whose mocked `msvcrt.locking` rejects `LK_NBLCK`.
- [ ] Run `test_ownership.py`; verify raw `PermissionError` escapes.
- [ ] Replace the pre-lock read with an end-position size check, retaining sentinel creation and nonblocking byte locking.
- [ ] Run `test_ownership.py`; verify the collision becomes `OwnershipError` and the handle closes.

### Task 2: Make duplicate GUI launch useful

**Files:**
- Modify: `tests/test_gui_app.py`
- Modify: `scripts/run_gui.py`

**Interfaces:**
- Consumes: initial `SupervisorInstanceLock.acquire()`.
- Produces: return code `0` and optional `webbrowser.open("http://127.0.0.1:<port>/")` on duplicate ownership.

- [ ] Write failing tests for normal and `--no-browser` duplicate launches.
- [ ] Run the focused tests; verify `OwnershipError` currently escapes.
- [ ] Acquire the instance lock explicitly before entering its context; on collision log the existing GUI URL, optionally open it, and return `0`.
- [ ] Run focused and full ownership/GUI tests.

### Task 3: Document, publish, and refresh review GUI

**Files:**
- Modify: `docs/user-guide.md`
- Modify: `AI_DEVELOPMENT_RULES.md`

**Interfaces:**
- Produces: durable startup semantics and a current review-only GUI process.

- [ ] Document duplicate-launch behavior and troubleshooting.
- [ ] Run the full suite, compile checks, and `git diff --check`.
- [ ] Commit and push `main`.
- [ ] Reconfirm the current PID serves review-only routes and ComfyUI queue is empty; stop only that PID and start updated `scripts/run_gui.py`.
- [ ] Verify acceptance route and assembled endpoint respond, `/api/status` remains absent while automatic rollout is locked, and exactly one GUI process owns the lock.
