"""FastAPI application for human review and remake batching."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .comfy_http import ComfyHttpError
from .gui_service import SupervisorController
from .review import (
    ReviewValidationError,
    scene_review_document,
    validate_scene_edit,
)
from .state_store import (
    RemakeMode,
    SceneRecord,
    SceneRevision,
    StateTransitionError,
)
from .storage import StorageLayout, write_json_atomic


def _record_document(record: SceneRecord) -> dict[str, Any]:
    result = asdict(record)
    result["state"] = record.state.value
    return result


def _revision_document(revision: SceneRevision) -> dict[str, Any]:
    result = asdict(revision)
    result["remake_mode"] = revision.remake_mode.value
    result["state"] = revision.state.value
    result["parameters"] = revision.parameters
    result["frame_url"] = (
        f"/api/media/{revision.job_id}/{revision.scene_id}/{revision.revision}/frame"
        if revision.frame_path
        else None
    )
    result["video_url"] = (
        f"/api/media/{revision.job_id}/{revision.scene_id}/{revision.revision}/video"
        if revision.video_path
        else None
    )
    return result


def _combo_values(document: Mapping[str, Any], node_type: str, input_name: str) -> list[str]:
    node = document.get(node_type)
    if not isinstance(node, Mapping):
        return []
    required = node.get("input", {}).get("required", {})
    definition = required.get(input_name) if isinstance(required, Mapping) else None
    if (
        isinstance(definition, list)
        and definition
        and isinstance(definition[0], list)
    ):
        return [str(item) for item in definition[0]]
    return []


def _materialize_original_revisions(
    controller: SupervisorController,
    storage: StorageLayout,
) -> None:
    for job_record in controller.store.list_jobs():
        job = controller.store.load_job(job_record.job_id)
        records = {item.scene_id: item for item in controller.store.scene_records(job.job_id)}
        for scene in job.scenes:
            record = records[scene.scene_id]
            document = scene_review_document(job, scene)
            controller.store.ensure_original_scene_revision(
                job.job_id,
                scene.scene_id,
                parameters=document,
                frame_path=record.frame_path,
                video_path=record.video_path,
            )
            manifest = storage.generation_manifest_path(job.job_id, scene.scene_id, 1)
            if not manifest.exists():
                write_json_atomic(
                    manifest,
                    {
                        "job_id": job.job_id,
                        "scene_id": scene.scene_id,
                        "revision": 1,
                        "remake_mode": RemakeMode.IMAGE_AND_VIDEO.value,
                        "parameters": document,
                        "frame_path": record.frame_path,
                        "video_path": record.video_path,
                        "status": record.state.value,
                        "imported_from_legacy_history": True,
                    },
                )


def create_gui_app(
    controller: SupervisorController,
    storage: StorageLayout,
    project_root: str | Path,
) -> FastAPI:
    project_root = Path(project_root)
    web_root = project_root / "web"
    _materialize_original_revisions(controller, storage)
    app = FastAPI(
        title="10MinVideoMaker Supervisor",
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return controller.status_document()

    @app.get("/api/jobs")
    async def jobs() -> list[dict[str, Any]]:
        return [
            {
                **asdict(item),
                "status": item.status.value,
            }
            for item in controller.store.list_jobs()
        ]

    @app.get("/api/jobs/{job_id}")
    async def job_detail(job_id: str) -> dict[str, Any]:
        try:
            job = controller.store.load_job(job_id)
            records = controller.store.scene_records(job_id)
        except StateTransitionError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {
            "job_id": job.job_id,
            "character": {
                "name": job.character.name,
                "series": job.character.series,
                "base_model": job.character.base_model,
            },
            "metadata": {
                key: value
                for key, value in job.raw.items()
                if key not in {"job_id", "character", "ltxv_character_lora", "scenes"}
            },
            "scenes": [
                {
                    **_record_document(record),
                    "title": next(
                        scene.title
                        for scene in job.scenes
                        if scene.scene_id == record.scene_id
                    ),
                    "revision_count": len(
                        controller.store.scene_revisions(job_id, record.scene_id)
                    ),
                }
                for record in records
            ],
        }

    @app.get("/api/jobs/{job_id}/scenes/{scene_id}")
    async def scene_detail(job_id: str, scene_id: int) -> dict[str, Any]:
        try:
            job = controller.store.load_job(job_id)
            scene = next(item for item in job.scenes if item.scene_id == scene_id)
            record = next(
                item
                for item in controller.store.scene_records(job_id)
                if item.scene_id == scene_id
            )
        except (StateTransitionError, StopIteration) as error:
            raise HTTPException(status_code=404, detail="Scene was not found.") from error
        revisions = controller.store.scene_revisions(job_id, scene_id)
        return {
            "parameters": scene_review_document(job, scene),
            "record": _record_document(record),
            "revisions": [_revision_document(item) for item in revisions],
        }

    @app.post("/api/jobs/{job_id}/approve")
    async def approve_job(job_id: str) -> dict[str, Any]:
        try:
            controller.approve_job(job_id)
        except StateTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"approved": True, "job_id": job_id}

    @app.get("/api/remake-batches")
    async def remake_batches() -> list[dict[str, Any]]:
        return [
            {
                **asdict(item),
                "state": item.state.value,
            }
            for item in controller.store.list_remake_batches()
        ]

    @app.post("/api/remake-batches")
    async def create_remake_batch(request: Request) -> dict[str, Any]:
        body = await request.json()
        items = body.get("items") if isinstance(body, Mapping) else None
        if not isinstance(items, list) or not items:
            raise HTTPException(status_code=400, detail="items must be a non-empty list.")
        validated: list[tuple[str, int, RemakeMode, Mapping[str, Any]]] = []
        try:
            for index, item in enumerate(items):
                if not isinstance(item, Mapping):
                    raise ReviewValidationError(f"items[{index}] must be an object.")
                job_id = str(item.get("job_id") or "")
                scene_id = item.get("scene_id")
                if isinstance(scene_id, bool) or not isinstance(scene_id, int):
                    raise ReviewValidationError(
                        f"items[{index}].scene_id must be an integer."
                    )
                mode = RemakeMode(str(item.get("remake_mode") or ""))
                edit = validate_scene_edit(
                    controller.store.load_job(job_id),
                    scene_id,
                    item.get("parameters"),
                )
                validated.append((job_id, scene_id, mode, edit.document))
            batch_id, revisions = controller.store.create_remake_batch(validated)
        except (ReviewValidationError, StateTransitionError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "batch_id": batch_id,
            "items": [
                {"job_id": job_id, "scene_id": scene_id, "revision": revision}
                for job_id, scene_id, revision in revisions
            ],
            "active_render": controller.active_render(),
        }

    @app.post("/api/remake-batches/{batch_id}/submit")
    async def submit_remake_batch(batch_id: str, request: Request) -> dict[str, Any]:
        body = await request.json()
        policy = body.get("collision_policy") if isinstance(body, Mapping) else None
        if controller.active_render() and policy not in {
            "after_current",
            "interrupt_current",
        }:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "An automated job is currently rendering.",
                    "choices": ["after_current", "interrupt_current"],
                },
            )
        policy = str(policy or "after_current")
        try:
            controller.queue_batch(batch_id, policy)
        except (StateTransitionError, ComfyHttpError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"submitted": True, "batch_id": batch_id, "collision_policy": policy}

    @app.get("/api/options")
    async def generation_options() -> dict[str, Any]:
        try:
            sampler_select = controller.supervisor.comfy.object_info("KSamplerSelect")
            sampler = controller.supervisor.comfy.object_info("KSampler")
        except ComfyHttpError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        samplers = _combo_values(sampler_select, "KSamplerSelect", "sampler_name")
        if not samplers:
            samplers = _combo_values(sampler, "KSampler", "sampler_name")
        return {
            "samplers": samplers,
            "schedulers": _combo_values(sampler, "KSampler", "scheduler"),
        }

    @app.get("/api/media/{job_id}/{scene_id}/{revision}/{kind}")
    async def scene_media(
        job_id: str,
        scene_id: int,
        revision: int,
        kind: str,
    ) -> FileResponse:
        if kind not in {"frame", "video"}:
            raise HTTPException(status_code=404, detail="Unknown media kind.")
        selected = next(
            (
                item
                for item in controller.store.scene_revisions(job_id, scene_id)
                if item.revision == revision
            ),
            None,
        )
        if selected is None:
            raise HTTPException(status_code=404, detail="Revision was not found.")
        raw_path = selected.frame_path if kind == "frame" else selected.video_path
        if not raw_path:
            raise HTTPException(status_code=404, detail="Media has not been generated.")
        path = Path(raw_path).resolve()
        try:
            path.relative_to(storage.root.resolve())
        except ValueError as error:
            raise HTTPException(status_code=403, detail="Media path is outside project storage.") from error
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Media file is missing.")
        return FileResponse(
            path,
            media_type="image/png" if kind == "frame" else "video/mp4",
            filename=path.name,
        )

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        async def stream():
            while True:
                yield "data: " + json.dumps(controller.status_document()) + "\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(stream(), media_type="text/event-stream")

    app.mount("/", StaticFiles(directory=web_root, html=True), name="web")
    return app
