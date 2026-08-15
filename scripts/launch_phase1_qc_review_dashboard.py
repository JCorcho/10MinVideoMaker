"""Launch a tiny local read-only review dashboard for Phase-1 canary QC artifacts."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote_plus, urlparse

from tenminvideomaker.configuration import load_project_environment
from tenminvideomaker.state_store import PipelineStateStore
from tenminvideomaker.storage import StorageLayout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_IDS = tuple(range(1, 29))
CANARY_DATABASE = Path("state") / "pipeline.sqlite3"


def _flatten_evaluation_summary(candidate: Any, store: PipelineStateStore) -> dict[str, Any]:
    latest = None
    evaluations = list(store.qc_evaluations(candidate.candidate_id))
    for item in evaluations:
        if item.state != "COMPLETE":
            continue
        latest = item
    if latest is None:
        return {
            "decision": None,
            "categories": [],
            "severity": None,
            "confidence": None,
            "timestamps": [],
            "strong_window_count": None,
            "started_at": None,
            "completed_at": None,
            "is_refusal": False,
            "infrastructure": None,
        }
    categories: set[str] = set()
    severities: list[int] = []
    confidences: list[float] = []
    times: list[str] = []
    is_refusal = bool(latest.raw_result and "refuse" in latest.raw_result.lower())
    for window in latest.suspect_windows:
        for defect in window.get("response", {}).get("errors", []):
            category = str(defect.get("category", "")).strip()
            if category:
                categories.add(category)
            severity = defect.get("severity")
            if isinstance(severity, (int, float)):
                severities.append(int(severity))
            confidence = defect.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))
            start = defect.get("start_time_seconds")
            end = defect.get("end_time_seconds")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                times.append(f"{start:.2f}-{end:.2f}s")
    decision = latest.normalized_decision.value if latest.normalized_decision is not None else None
    confidence = None
    severity = None
    if confidences:
        confidence = round(sum(confidences) / len(confidences), 4)
    if severities:
        severity = max(severities)
    return {
        "decision": decision,
        "categories": sorted(categories),
        "severity": severity,
        "confidence": confidence,
        "timestamps": times[:20],
        "strong_window_count": latest.strong_window_count,
        "started_at": latest.started_at,
        "completed_at": latest.completed_at,
        "is_refusal": is_refusal,
        "infrastructure": latest.raw_result if latest.raw_result is not None else None,
    }


def collect_review_payload(canary_roots: tuple[Path, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {"scenes": {}, "canary_roots": []}
    for root in canary_roots:
        db = root / CANARY_DATABASE
        if not db.is_file():
            raise RuntimeError(f"Canary DB missing: {db}")
        store = PipelineStateStore(db)
        store.initialize()
        snapshot = store.snapshot()
        job_id = snapshot.job_id
        if job_id is None:
            raise RuntimeError(f"No claimed job in {root}")

        scenes: dict[int, dict[str, Any]] = {scene_id: {} for scene_id in SCENE_IDS}
        for candidate in store.qc_candidates(job_id):
            if candidate.scene_id < 1 or candidate.scene_id > 28:
                continue
            row = {
                "candidate_id": candidate.candidate_id,
                "tier": candidate.tier.value,
                "revision": candidate.revision,
                "state": candidate.state.value,
                "next_action": candidate.next_action,
                "video_path": str(candidate.source_video_path),
                "infrastructure_failure_count": candidate.infrastructure_failure_count,
                "infrastructure_failure": "",
            }
            failure = candidate.last_failure
            if isinstance(failure, Mapping) and failure.get("message"):
                row["infrastructure_failure"] = str(failure["message"]).strip()
            row.update(_flatten_evaluation_summary(candidate, store))
            scenes[candidate.scene_id][candidate.tier.value] = row

        payload["scenes"].update(scenes)
        payload["canary_roots"].append(
            {
                "root": str(root.resolve()),
                "job_id": job_id,
                "scenes_count": len(store.scene_records(job_id)),
            }
        )
    payload["canary_roots"] = sorted(payload["canary_roots"], key=lambda item: str(item["root"]))
    return payload


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Phase-1 QC Review</title>
  <style>
    body { font-family: 'Inter', 'Arial', sans-serif; margin: 12px; background: #f3f5f7; color: #19202a; }
    .panel { background: #fff; border: 1px solid #cbd4dd; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .top { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
    select { font-size: 1rem; }
    .tier { border: 1px dashed #cfd8e2; border-radius: 8px; padding: 8px; margin: 8px 0; }
    video { width: 100%; max-width: 100%; background: #000; border-radius: 6px; }
    .label { font-weight: 700; margin-top: 4px; }
    .annotation button { margin-right: 8px; }
    .infra { color: #7a1f1f; }
    code { word-break: break-all; }
  </style>
</head>
<body>
  <div class='panel'>
    <h2 style='margin:0'>Phase-1 QC Review Dashboard</h2>
    <p style='margin:6px 0; color:#435062;'>Read-only local review for canary QC candidates only.</p>
    <div class='top'>
      <label for='scene-select'>Scene:</label>
      <select id='scene-select'></select>
      <span id='scene-title'></span>
    </div>
    <div id='scene-summary'></div>
  </div>
  <div id='scene-containers'></div>

  <script>
    const data = __PAYLOAD__;
    const scenes = data.scenes || {};
    const selector = document.getElementById('scene-select');
    const containers = document.getElementById('scene-containers');
    const title = document.getElementById('scene-title');
    const summary = document.getElementById('scene-summary');
    const annotations = JSON.parse(localStorage.getItem('phase1_qc_annotations') || '{}');

    function escapeHtml(input) {
      return String(input).replace(/[&<>\"']/g, (c) => {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c];
      });
    }

    function renderSceneSelect() {
      selector.innerHTML = '';
      Object.keys(scenes).sort((a, b) => Number(a) - Number(b)).forEach((sceneId) => {
        const option = document.createElement('option');
        option.value = sceneId;
        option.textContent = `Scene ${sceneId}`;
        selector.appendChild(option);
      });
      selector.addEventListener('change', () => renderScene(selector.value));
    }

    function infraSummary(row) {
      if (!row || !row.infrastructure_failure_count) return '';
      const message = row.infrastructure_failure || 'No message';
      return `<div class='infra'>Infrastructure failure (${row.infrastructure_failure_count}): ${escapeHtml(message)}</div>`;
    }

    function setAnnotation(sceneId, tier, value) {
      const key = `${sceneId}::${tier}`;
      if (!value) {
        delete annotations[key];
      } else {
        annotations[key] = value;
      }
      localStorage.setItem('phase1_qc_annotations', JSON.stringify(annotations));
      renderScene(sceneId);
    }

    function decisionHtml(row) {
      const decision = row.decision || 'NONE';
      const color = decision === 'PASS' ? '#1d5f1d' : decision === 'FAIL' ? '#8f1f1f' : '#775b1c';
      return `<div><span class='label'>Decision:</span> <span style='color:${color}'>${escapeHtml(decision)}</span></div>`;
    }

    function renderCandidateCard(sceneId, tier, row) {
      if (!row || !row.video_path) {
        return `<div class='tier'><strong>${tier}</strong><div>Not available.</div></div>`;
      }
      const key = `${sceneId}::${tier}`;
      const current = annotations[key] || '—';
      const pathDisplay = escapeHtml(row.video_path);
      const source = `/media?path=${encodeURIComponent(row.video_path)}`;
      const categories = (row.categories || []).join(', ') || '—';
      const infra = infraSummary(row);
      const annotation = `<div><span class='label'>Human annotation:</span> <strong>${escapeHtml(current)}</strong></div>`;
      const infraText = infra ? `<div>${infra}</div>` : '';
      const infraReason = row.infrastructure_failure ? escapeHtml(row.infrastructure_failure) : '';
      return `
        <div class='tier'>
          <div class='label'>${tier} (rev ${row.revision})</div>
          <div>Candidate: ${escapeHtml(row.candidate_id)}</div>
          <div>State: ${escapeHtml(row.state || '')}</div>
          <div>Next action: ${escapeHtml(row.next_action || '')}</div>
          ${decisionHtml(row)}
          <div>Categories: ${categories}</div>
          <div>Max severity: ${row.severity !== null ? row.severity : '—'}</div>
          <div>Mean confidence: ${row.confidence !== null ? row.confidence : '—'}</div>
          <div>Infrastructure failures: ${row.infrastructure_failure_count || 0}</div>
          <div>Strong windows: ${row.strong_window_count ?? '—'}</div>
          <div>Timestamps: ${(row.timestamps || []).join(', ') || '—'}</div>
          <div>Path: <code>${pathDisplay}</code></div>
          <video controls preload='metadata'>
            <source src='${source}' type='video/mp4' />
            Your browser cannot play this clip.
          </video>
          ${infraText}
          ${infraReason ? `<div class='infra'>${infraReason}</div>` : ''}
          <div class='annotation'>
            <div class='label'>Local notes:</div>
            <button onclick="setAnnotation('${sceneId}','${tier}','HUMAN LOOKS GOOD')">HUMAN LOOKS GOOD</button>
            <button onclick="setAnnotation('${sceneId}','${tier}','HUMAN LOOKS BAD')">HUMAN LOOKS BAD</button>
            <button onclick="setAnnotation('${sceneId}','${tier}','UNSURE')">UNSURE</button>
            <button onclick="setAnnotation('${sceneId}','${tier}','')">Clear</button>
            ${annotation}
          </div>
        </div>
      `;
    }

    function renderScene(sceneId) {
      const scene = scenes[sceneId] || {};
      title.textContent = `Review set for scene ${sceneId}`;
      const canaries = data.canary_roots.map((item) => item.root).join(', ');
      summary.innerHTML = `<div class='panel'>Candidate roots: ${escapeHtml(canaries)}</div>`;
      containers.innerHTML = ['ORIGINAL', 'A1', 'B1'].map((tier) => renderCandidateCard(sceneId, tier, scene[tier])).join('');
    }

    function run() {
      renderSceneSelect();
      selector.value = '1';
      renderScene('1');
    }

    run();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, payload: Mapping[str, Any], canary_roots: tuple[Path, ...]) -> None:
        self._payload = payload
        self._canary_roots = canary_roots
        super().__init__(*args)

    def _json(self, status_code: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        path: Path,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        payload = path.read_bytes()
        file_size = len(payload)
        content_type, _ = mimetypes.guess_type(str(path))
        if content_type is None:
            content_type = "application/octet-stream"
        partial = start is not None or end is not None
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            range_end = file_size - 1 if end is None else min(end, file_size - 1)
            range_start = 0 if start is None else start
            self.send_header("Content-Range", f"bytes {range_start}-{range_end}/{file_size}")
            self.send_header("Content-Length", str(range_end - range_start + 1))
            self.end_headers()
            self.wfile.write(payload[range_start : range_end + 1])
        else:
            self.send_header("Content-Length", str(file_size))
            self.end_headers()
            self.wfile.write(payload)

    def _absolute_canary_root(self, value: str) -> Path | None:
        candidate = Path(value).resolve()
        for root in self._canary_roots:
            try:
                candidate.relative_to(root.resolve())
                return candidate
            except ValueError:
                continue
        return None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/data":
            self._json(200, self._payload)
            return
        if parsed.path == "/media":
            query = parse_qs(parsed.query)
            encoded = query.get("path", [""])[0]
            requested = Path(unquote_plus(encoded))
            if not requested.is_absolute():
                requested = Path(requested).resolve()
            requested = self._absolute_canary_root(str(requested))
            if requested is None or not requested.is_file():
                self.send_error(404, "Missing media file")
                return
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                range_spec = range_header.split("=", 1)[1]
                start_s, _, end_s = range_spec.partition("-")
                start = int(start_s) if start_s else None
                end = int(end_s) if end_s else None
                self._send_bytes(requested, start=start, end=end)
                return
            self._send_bytes(requested)
            return
        if parsed.path in {"", "/"}:
            html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(self._payload))
            encoded = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        self.send_error(404, "Not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canary-root",
        action="append",
        type=Path,
        required=True,
        help="Canary root to include in dashboard.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canary_roots = tuple(sorted({Path(item).resolve() for item in args.canary_root}, key=lambda item: str(item)))
    payload = collect_review_payload(canary_roots)
    load_project_environment(PROJECT_ROOT, storage_layout=StorageLayout.configured())
    server = ThreadingHTTPServer(
        (args.host, args.port),
        lambda *inner_args: DashboardHandler(
            *inner_args,
            payload=payload,
            canary_roots=canary_roots,
        ),
    )
    url = f"http://{args.host}:{server.server_port}"
    print(f"Phase-1 human review dashboard: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
