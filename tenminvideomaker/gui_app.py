"""FastAPI application for human review and remake batching."""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import secrets
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .acceptance_review import (
    AcceptanceReviewError,
    AcceptanceReviewProxyError,
    AcceptanceReviewService,
)
from .comfy_http import ComfyHttpError
from .gui_service import GuiServiceError, SupervisorController
from .review import (
    ReviewValidationError,
    scene_review_document,
    validate_scene_edit,
)
from .state_store import (
    RemakeMode,
    ManualFinalRecord,
    SceneRecord,
    SceneRevision,
    StateTransitionError,
)
from .qc_contracts import QcHumanDecision
from .storage import StorageLayout, write_json_atomic


def _project_date(job: Any, stored_created_at: str | None = None) -> str:
    source_created_at = job.raw.get("created_at")
    if isinstance(source_created_at, str):
        try:
            return datetime.fromisoformat(
                source_created_at.replace("Z", "+00:00")
            ).strftime("%m/%d/%Y")
        except ValueError:
            pass
    if len(job.job_id) >= 8 and job.job_id[:8].isdigit():
        try:
            return datetime.strptime(job.job_id[:8], "%Y%m%d").strftime("%m/%d/%Y")
        except ValueError:
            pass
    if stored_created_at:
        try:
            return datetime.fromisoformat(
                stored_created_at.replace("Z", "+00:00")
            ).strftime("%m/%d/%Y")
        except ValueError:
            pass
    return "Unknown date"


def _project_display_name(job: Any, stored_created_at: str | None = None) -> str:
    return f"{job.character.name} · {_project_date(job, stored_created_at)}"


def _record_document(record: SceneRecord) -> dict[str, Any]:
    result = asdict(record)
    result["state"] = record.state.value
    return result


def _revision_document(
    revision: SceneRevision,
    controller: SupervisorController | None = None,
) -> dict[str, Any]:
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
    result["chunk_progress"] = (
        controller.chunk_progress_document(
            revision.job_id,
            revision.scene_id,
            revision.revision,
        )
        if controller is not None
        else None
    )
    return result


def _manual_final_document(record: ManualFinalRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "request_id": record.request_id,
        "job_id": record.job_id,
        "state": record.state.value,
        "error": record.error,
        "output_available": bool(record.output_path),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


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


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(values), key=str.casefold)


def _is_loopback_client(host: str | None) -> bool:
    return host in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


def _lan_request_authorized(request: Request, password: str) -> bool:
    if _is_loopback_client(request.client.host if request.client else None):
        return True
    authorization = request.headers.get("authorization", "")
    scheme, _, encoded = authorization.partition(" ")
    if scheme.casefold() != "basic" or not encoded:
        return False
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    username, separator, supplied_password = decoded.partition(":")
    return bool(separator) and secrets.compare_digest(username, "10min") and secrets.compare_digest(
        supplied_password,
        password,
    )


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


def _configure_lan_auth(app: FastAPI, lan_password: str | None) -> None:
    if not lan_password:
        return

    @app.middleware("http")
    async def require_lan_password(request: Request, call_next: Any) -> Response:
        if not _lan_request_authorized(request, lan_password):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="10MinVideoMaker LAN"'},
            )
        return await call_next(request)


def _register_acceptance_review_routes(
    app: FastAPI,
    acceptance_review: AcceptanceReviewService,
) -> None:
    @app.get("/api/acceptance-runs")
    def acceptance_runs() -> list[dict[str, object]]:
        return acceptance_review.list_runs()

    @app.get("/api/acceptance-runs/{run_id}")
    def acceptance_run(run_id: str) -> dict[str, object]:
        try:
            return acceptance_review.review_document(run_id)
        except AcceptanceReviewError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/acceptance-runs/{run_id}/media/{role}")
    def acceptance_media(run_id: str, role: str) -> FileResponse:
        try:
            path = acceptance_review.review_proxy_path(run_id, role)
        except AcceptanceReviewProxyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except AcceptanceReviewError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/api/acceptance-runs/{run_id}/assembled/{case_name}")
    def acceptance_assembled(run_id: str, case_name: str) -> FileResponse:
        try:
            path = acceptance_review.assembled_proxy_path(run_id, case_name)
        except AcceptanceReviewProxyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except AcceptanceReviewError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/api/acceptance-runs/{run_id}/stills/{case_name}/{asset_name}")
    def acceptance_still(
        run_id: str,
        case_name: str,
        asset_name: str,
    ) -> FileResponse:
        try:
            path = acceptance_review.still_path(run_id, case_name, asset_name)
        except AcceptanceReviewError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(path, media_type="image/png", filename=path.name)


def create_acceptance_review_app(
    storage: StorageLayout,
    project_root: str | Path,
    *,
    lan_password: str | None = None,
) -> FastAPI:
    """Serve bounded continuation review without starting pipeline control."""
    project_root = Path(project_root)
    app = FastAPI(
        title="10MinVideoMaker Continuation Review",
        docs_url=None,
        redoc_url=None,
    )
    _configure_lan_auth(app, lan_password)
    _register_acceptance_review_routes(app, AcceptanceReviewService(storage))

    @app.get("/", include_in_schema=False)
    def review_root() -> RedirectResponse:
        return RedirectResponse("/acceptance-review.html")

    app.mount(
        "/",
        StaticFiles(directory=project_root / "web", html=True),
        name="acceptance-review-web",
    )
    return app


def create_gui_app(
    controller: SupervisorController,
    storage: StorageLayout,
    project_root: str | Path,
    *,
    lan_password: str | None = None,
) -> FastAPI:
    project_root = Path(project_root)
    web_root = project_root / "web"
    _materialize_original_revisions(controller, storage)
    app = FastAPI(
        title="10MinVideoMaker Supervisor",
        docs_url=None,
        redoc_url=None,
    )
    acceptance_review = AcceptanceReviewService(storage)

    _configure_lan_auth(app, lan_password)

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return controller.status_document()

    @app.get("/api/jobs")
    async def jobs() -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for item in controller.store.list_jobs():
            job = controller.store.load_job(item.job_id)
            documents.append(
                {
                    **asdict(item),
                    "status": item.status.value,
                    "display_name": _project_display_name(job, item.created_at),
                }
            )
        return documents

    @app.get("/api/qc/review-queue")
    async def qc_review_queue() -> dict[str, Any]:
        return controller.qc_review_queue_document()

    @app.post(
        "/api/qc/jobs/{job_id}/scenes/{scene_id}/candidates/{candidate_id}/decision"
    )
    async def qc_human_decision(
        job_id: str,
        scene_id: int,
        candidate_id: str,
        request: Request,
    ) -> dict[str, Any]:
        body = await request.json()
        if not isinstance(body, Mapping) or set(body) != {"decision"}:
            raise HTTPException(
                status_code=400,
                detail="Payload must contain only the durable QC decision.",
            )
        try:
            decision = QcHumanDecision(str(body["decision"]))
            if decision not in {
                QcHumanDecision.APPROVE,
                QcHumanDecision.REJECT,
                QcHumanDecision.HOLD,
            }:
                raise ValueError
            result = controller.decide_qc_candidate(
                job_id=job_id,
                scene_id=scene_id,
                candidate_id=candidate_id,
                decision=decision,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Unknown QC decision.") from error
        except StateTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "job_id": job_id,
            "scene_id": scene_id,
            "candidate_id": candidate_id,
            "decision_id": result.decision.decision_id,
            "decision": result.decision.decision.value,
            "candidate_state": result.candidate.state.value,
            "replayed": result.replayed,
        }

    @app.get("/api/jobs/{job_id}")
    async def job_detail(job_id: str) -> dict[str, Any]:
        try:
            job = controller.store.load_job(job_id)
            records = controller.store.scene_records(job_id)
            job_record = next(
                item
                for item in controller.store.list_jobs()
                if item.job_id == job_id
            )
        except StateTransitionError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        scenes: list[dict[str, Any]] = []
        for record in records:
            revisions = controller.store.scene_revisions(job_id, record.scene_id)
            latest = revisions[0] if revisions else None
            scenes.append(
                {
                    **_record_document(record),
                    "title": next(
                        scene.title
                        for scene in job.scenes
                        if scene.scene_id == record.scene_id
                    ),
                    "revision_count": len(revisions),
                    "chunk_progress": (
                        controller.chunk_progress_document(
                            job_id,
                            record.scene_id,
                            latest.revision,
                        )
                        if latest is not None
                        else None
                    ),
                }
            )
        return {
            "job_id": job.job_id,
            "display_name": _project_display_name(job, job_record.created_at),
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
            "scenes": scenes,
            "manual_final": _manual_final_document(
                controller.store.latest_manual_final(job_id)
            ),
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
            "revisions": [
                _revision_document(item, controller) for item in revisions
            ],
        }

    @app.post("/api/jobs/{job_id}/approve")
    async def approve_job(job_id: str) -> dict[str, Any]:
        try:
            controller.approve_job(job_id)
        except StateTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"approved": True, "job_id": job_id}

    @app.post("/api/pipeline/cancel-current")
    async def cancel_current_project() -> dict[str, Any]:
        """Cancel the held automatic project and free the pipeline for the next job."""
        try:
            result = controller.cancel_current_project()
        except GuiServiceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except StateTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"cancelled": True, **result}

    @app.put("/api/jobs/{job_id}/scenes/{scene_id}/manual-final-inclusion")
    async def set_manual_final_inclusion(
        job_id: str,
        scene_id: int,
        request: Request,
    ) -> dict[str, Any]:
        body = await request.json()
        included = body.get("included") if isinstance(body, Mapping) else None
        if not isinstance(included, bool):
            raise HTTPException(status_code=400, detail="included must be a boolean.")
        try:
            controller.store.set_scene_manual_final_inclusion(
                job_id,
                scene_id,
                included=included,
            )
        except StateTransitionError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {
            "job_id": job_id,
            "scene_id": scene_id,
            "include_in_manual_final": included,
        }

    @app.post("/api/jobs/{job_id}/manual-final")
    async def queue_manual_final(job_id: str) -> dict[str, Any]:
        try:
            request = controller.queue_manual_final(job_id)
        except StateTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return _manual_final_document(request) or {}

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
            t2i_loader = controller.supervisor.comfy.object_info("LoraLoader")
            i2v_loader = controller.supervisor.comfy.object_info("LoraLoaderModelOnly")
        except ComfyHttpError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        samplers = _combo_values(sampler_select, "KSamplerSelect", "sampler_name")
        if not samplers:
            samplers = _combo_values(sampler, "KSampler", "sampler_name")
        return {
            "samplers": _unique_sorted(samplers),
            "schedulers": _unique_sorted(_combo_values(sampler, "KSampler", "scheduler")),
            "t2i_loras": _unique_sorted(
                _combo_values(t2i_loader, "LoraLoader", "lora_name")
            ),
            "i2v_loras": _unique_sorted(
                _combo_values(i2v_loader, "LoraLoaderModelOnly", "lora_name")
            ),
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

    _register_acceptance_review_routes(app, acceptance_review)
    app.mount("/", StaticFiles(directory=web_root, html=True), name="web")
    return app
