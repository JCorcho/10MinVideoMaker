"""Archive and requeue interrupted v1 continuation scenes for the v2 strategy."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.comfy_http import ComfyHttpClient
from tenminvideomaker.continuation import LEGACY_CONTINUATION_STRATEGIES
from tenminvideomaker.ownership import SupervisorInstanceLock
from tenminvideomaker.state_store import PipelineStateStore, StateTransitionError
from tenminvideomaker.storage import StorageLayout, write_json_atomic


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Archive immutable ltx23_latent_overlap_v1 artifacts, preserve their "
            "audit snapshot, and requeue those original scenes for the current "
            "exact-frame strategy. Dry-run is the default."
        )
    )
    parser.add_argument("--job-id", help="Defaults to the active saved job.")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--apply", action="store_true")
    return parser


def legacy_snapshots(
    store: PipelineStateStore,
    job_id: str,
) -> tuple[Mapping[str, Any], ...]:
    snapshots: list[Mapping[str, Any]] = []
    for scene in store.scene_records(job_id):
        plan = store.continuation_plan(job_id, scene.scene_id, 1)
        if plan is None or plan.plan.get("strategy") not in LEGACY_CONTINUATION_STRATEGIES:
            continue
        snapshots.append(
            store.continuation_revision_snapshot(job_id, scene.scene_id, 1)
        )
    return tuple(snapshots)


def _move_if_present(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"Upgrade archive destination already exists: {destination}")
    shutil.move(str(source), str(destination))
    return True


def archive_and_reset(
    store: PipelineStateStore,
    storage: StorageLayout,
    snapshot: Mapping[str, Any],
) -> Path:
    job_id = str(snapshot["job_id"])
    scene_id = int(snapshot["scene_id"])
    revision = int(snapshot["revision"])
    plan = snapshot["plan"]
    plan_hash = str(plan["plan_hash"])
    strategy = str(plan["document"]["strategy"])
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    revision_root = storage.revision_root(job_id, scene_id, revision)
    archive_root = (
        revision_root
        / "history"
        / f"continuation-upgrade-{stamp}-{plan_hash[:12]}"
    )
    archive_root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(archive_root / "sqlite-audit-snapshot.json", snapshot)

    move_pairs = (
        (revision_root / "chunks", archive_root / "chunks"),
        (storage.scene_assembly_root(job_id, scene_id, revision), archive_root / "assembly"),
        (storage.scene_clip_path(job_id, scene_id, revision), archive_root / "video.mp4"),
        (
            storage.generation_manifest_path(job_id, scene_id, revision),
            archive_root / "generation-manifest.json",
        ),
    )
    moved: list[tuple[Path, Path]] = []
    try:
        for source, destination in move_pairs:
            if _move_if_present(source, destination):
                moved.append((source, destination))
        store.reset_legacy_original_continuation(
            job_id,
            scene_id,
            expected_plan_hash=plan_hash,
            expected_strategy=strategy,
        )
    except BaseException:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                shutil.move(str(destination), str(source))
        raise
    write_json_atomic(
        archive_root / "UPGRADED.json",
        {
            "job_id": job_id,
            "scene_id": scene_id,
            "revision": revision,
            "archived_strategy": strategy,
            "archived_plan_hash": plan_hash,
            "upgraded_at": datetime.now(UTC).isoformat(),
        },
    )
    return archive_root


def main() -> int:
    args = argument_parser().parse_args()
    storage = StorageLayout.configured()
    storage.ensure()
    store = PipelineStateStore(storage.database_path)
    active = store.snapshot()
    job_id = args.job_id or active.job_id
    if not job_id:
        raise SystemExit("No active saved job and no --job-id was supplied.")
    snapshots = legacy_snapshots(store, job_id)
    if not snapshots:
        print(f"No legacy continuation plans found for job {job_id}.")
        return 0
    scenes = ", ".join(str(item["scene_id"]) for item in snapshots)
    print(f"Legacy continuation scenes for {job_id}: {scenes}")
    if not args.apply:
        print("Dry-run only. Re-run with --apply after stopping the project supervisor.")
        return 0

    with SupervisorInstanceLock(storage.instance_lock_path):
        running, pending = ComfyHttpClient(args.comfy_url).queue_counts()
        if running or pending:
            raise SystemExit(
                "ComfyUI queue is not empty; legacy continuation state was not changed."
            )
        for snapshot in snapshots:
            archive = archive_and_reset(store, storage, snapshot)
            print(f"Archived and requeued scene {snapshot['scene_id']}: {archive}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, StateTransitionError) as error:
        raise SystemExit(f"Legacy continuation upgrade failed: {error}") from error
