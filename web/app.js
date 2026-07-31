const state = {
  jobs: [],
  scenes: [],
  selectedJob: null,
  selectedScene: null,
  sceneData: null,
  working: null,
  drafts: new Map(),
  revisionDrafts: new Map(),
  manualFinal: null,
  options: { samplers: [], schedulers: [], t2i_loras: [], i2v_loras: [] },
  pendingBatch: null,
  status: null,
  cancellingProject: false,
  activeChunkProgress: null,
  progressRefreshes: new Set(),
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const mobileQuery = window.matchMedia("(max-width: 760px)");

function mobileView() {
  if (!mobileQuery.matches) {
    document.body.removeAttribute("data-mobile-view");
    document.documentElement.style.removeProperty("--mobile-topbar-height");
    document.documentElement.style.removeProperty(
      "--mobile-scene-switcher-height",
    );
    return;
  }
  document.body.dataset.mobileView = !state.selectedJob
    ? "projects"
    : state.selectedScene == null
      ? "scenes"
      : "detail";
  document.documentElement.style.setProperty(
    "--mobile-topbar-height",
    `${$(".topbar").offsetHeight}px`,
  );
  const switcherHeight = $("#mobile-scene-switcher")?.offsetHeight || 76;
  document.documentElement.style.setProperty(
    "--mobile-scene-switcher-height",
    `${switcherHeight}px`,
  );
}

function scrollMobilePanel(selector) {
  if (!mobileQuery.matches) return;
  requestAnimationFrame(() => $(selector)?.scrollTo({ top: 0, behavior: "smooth" }));
}

function renderMobileScenePicker() {
  const select = $("#mobile-scene-select");
  const scenes = state.scenes || [];
  select.disabled = !scenes.length;
  select.innerHTML = [
    `<option value="" disabled ${state.selectedScene == null ? "selected" : ""}>Choose a scene…</option>`,
    ...scenes.map((scene) =>
      `<option value="${scene.scene_id}">${String(scene.scene_id).padStart(2, "0")} · ${escapeHtml(scene.title)}</option>`
    ),
  ].join("");
  if (state.selectedScene != null) select.value = String(state.selectedScene);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const body = response.headers.get("content-type")?.includes("application/json")
    ? await response.json()
    : null;
  if (!response.ok) {
    const detail = body?.detail;
    const message = typeof detail === "string" ? detail : detail?.message || `Request failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return body;
}

function clone(value) {
  return structuredClone(value);
}

function normalizeSceneParameters(value) {
  const document = clone(value);
  document.i2v ||= {};
  if (!Array.isArray(document.i2v.segments)) {
    document.i2v.segments = [];
  }
  document.i2v.segments.forEach((segment) => {
    if (
      segment
      && typeof segment === "object"
      && segment.new_transition_frames != null
    ) {
      delete segment.requested_duration_seconds;
    }
  });
  const continuation = document.i2v.temporal_continuation;
  if (continuation && typeof continuation === "object") {
    const legacyDuration = Number(continuation.requested_duration_seconds);
    if (Number.isFinite(legacyDuration) && legacyDuration > 0) {
      document.estimated_seconds = legacyDuration;
    }
    delete continuation.requested_duration_seconds;
  }
  return document;
}

function getPath(object, path) {
  return path.split(".").reduce((value, key) => value?.[key], object);
}

function setPath(object, path, value) {
  const parts = path.split(".");
  const last = parts.pop();
  const target = parts.reduce((value, key) => value[key], object);
  target[last] = value;
}

function humanize(key) {
  return String(key).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function displayValue(value) {
  if (value == null) return "None";
  if (Array.isArray(value)) return value.map(displayValue).join(", ");
  if (typeof value === "object") {
    return Object.entries(value).map(([key, item]) => `${humanize(key)}: ${displayValue(item)}`).join("\n");
  }
  return String(value);
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.add("hidden"), 4200);
}

function badge(value) {
  return `<span class="badge ${value}">${humanize(value)}</span>`;
}

function projectDate(job, detail) {
  const sourceDate = detail?.metadata?.created_at;
  if (typeof sourceDate === "string") {
    const parsed = new Date(sourceDate);
    if (!Number.isNaN(parsed.valueOf())) {
      return `${String(parsed.getUTCMonth() + 1).padStart(2, "0")}/${String(parsed.getUTCDate()).padStart(2, "0")}/${parsed.getUTCFullYear()}`;
    }
  }
  const encoded = String(job.job_id).match(/^(\d{4})(\d{2})(\d{2})/);
  if (encoded) {
    return `${encoded[2]}/${encoded[3]}/${encoded[1]}`;
  }
  const stored = new Date(job.created_at);
  if (!Number.isNaN(stored.valueOf())) {
    return `${String(stored.getUTCMonth() + 1).padStart(2, "0")}/${String(stored.getUTCDate()).padStart(2, "0")}/${stored.getUTCFullYear()}`;
  }
  return "Unknown date";
}

async function loadJobs() {
  const jobs = await api("/api/jobs");
  state.jobs = await Promise.all(jobs.map(async (job) => {
    if (job.display_name) return job;
    try {
      const detail = await api(`/api/jobs/${encodeURIComponent(job.job_id)}`);
      return {
        ...job,
        display_name: `${detail.character.name} · ${projectDate(job, detail)}`,
      };
    } catch {
      return job;
    }
  }));
  renderJobs();
  if (state.selectedJob) {
    await selectJob(state.selectedJob);
  } else {
    mobileView();
  }
}

function renderJobs() {
  $("#jobs").innerHTML = state.jobs.map((job) => `
    <button class="list-card ${job.job_id === state.selectedJob ? "selected" : ""}" data-job="${escapeHtml(job.job_id)}">
      <strong>${escapeHtml(job.display_name || job.job_id)}</strong>
      <div class="card-row">${badge(job.status)}<span>${job.succeeded_count}/${job.scene_count} scenes</span></div>
    </button>
  `).join("") || `<p class="muted">No stored projects yet.</p>`;
  $$("[data-job]").forEach((button) => button.addEventListener("click", () => selectJob(button.dataset.job)));
}

async function selectJob(jobId) {
  state.selectedJob = jobId;
  state.selectedScene = null;
  state.sceneData = null;
  state.scenes = [];
  state.manualFinal = null;
  renderJobs();
  renderMobileScenePicker();
  mobileView();
  const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
  const summary = state.jobs.find((item) => item.job_id === jobId);
  $("#job-title").textContent = job.display_name || summary?.display_name || job.job_id;
  $("#job-meta").textContent = `${job.character.name} · ${job.character.series} · ${job.character.base_model}`;
  state.scenes = job.scenes;
  state.manualFinal = job.manual_final;
  renderScenes();
  renderManualFinalControls();
  $("#approve-job").classList.toggle("hidden", state.status?.pipeline_state !== "awaiting_review" || state.status?.job_id !== jobId);
  renderMobileScenePicker();
  mobileView();
  scrollMobilePanel(".scenes-panel");
}

function renderScenes() {
  $("#scenes").innerHTML = state.scenes.map((scene) => `
    <button class="list-card ${scene.scene_id === state.selectedScene ? "selected" : ""}" data-scene="${scene.scene_id}">
      <strong>${String(scene.scene_id).padStart(2, "0")} · ${escapeHtml(scene.title)}</strong>
      <div class="card-row">${badge(scene.state)}<span>${scene.include_in_manual_final ? "included" : "excluded"} · ${scene.revision_count} version${scene.revision_count === 1 ? "" : "s"}</span></div>
      ${scene.chunk_progress ? `<span class="chunk-card-progress">Chunk ${scene.chunk_progress.current_chunk}/${scene.chunk_progress.total_chunks} · ${humanize(scene.chunk_progress.phase)}</span>` : ""}
    </button>
  `).join("");
  $$("[data-scene]").forEach((button) => button.addEventListener("click", () => selectScene(Number(button.dataset.scene))));
}

function renderManualFinalControls() {
  const controls = $("#manual-final-controls");
  const included = state.scenes.filter((scene) => scene.include_in_manual_final).length;
  controls.classList.toggle("hidden", !state.selectedJob);
  $("#manual-final-summary").textContent = `${included}/${state.scenes.length} scene${state.scenes.length === 1 ? "" : "s"} included · latest successful version of each will be used.`;
  const request = state.manualFinal;
  const active = ["queued", "running"].includes(request?.state);
  $("#render-project-final").disabled = !included || active;
  $("#manual-final-status").textContent = request
    ? `Last manual final: ${humanize(request.state)}${request.error ? ` · ${request.error}` : request.output_available ? " · output ready" : ""}`
    : "Automatic first-run concat is unchanged. This runs only when you press the button.";
}

async function selectScene(sceneId) {
  state.selectedScene = sceneId;
  mobileView();
  state.sceneData = await api(`/api/jobs/${encodeURIComponent(state.selectedJob)}/scenes/${sceneId}`);
  const key = `${state.selectedJob}:${sceneId}`;
  const draft = state.drafts.get(key);
  $("#empty-state").classList.add("hidden");
  $("#scene-detail").classList.remove("hidden");
  $$("[data-scene]").forEach((button) => button.classList.toggle("selected", Number(button.dataset.scene) === sceneId));
  renderRevisionPicker(draft?.source_revision);
  state.working = normalizeSceneParameters(
    revisionDraft(draft, selectedRevision())?.parameters
      || revisionParameters(selectedRevision()),
  );
  renderSceneHeading(selectedRevision());
  renderForm();
  $("#mark-remake").checked = Boolean(draft);
  $("#remake-mode").classList.toggle("hidden", !draft);
  if (draft) {
    $(`input[name="remake-mode"][value="${draft.remake_mode}"]`).checked = true;
  }
  $("#include-in-manual-final").checked = Boolean(
    state.sceneData.record.include_in_manual_final,
  );
  setEditable(Boolean(draft));
  renderMobileScenePicker();
  mobileView();
  scrollMobilePanel(".detail-panel");
}

function backToProjects() {
  state.selectedJob = null;
  state.selectedScene = null;
  state.sceneData = null;
  state.working = null;
  state.scenes = [];
  state.manualFinal = null;
  $("#job-title").textContent = "Choose a project";
  $("#job-meta").textContent = "";
  $("#scenes").innerHTML = "";
  renderManualFinalControls();
  $("#scene-detail").classList.add("hidden");
  $("#empty-state").classList.remove("hidden");
  renderJobs();
  renderMobileScenePicker();
  mobileView();
  scrollMobilePanel(".library-panel");
}

function backToScenes() {
  state.selectedScene = null;
  state.sceneData = null;
  state.working = null;
  $("#scene-detail").classList.add("hidden");
  $("#empty-state").classList.remove("hidden");
  renderMobileScenePicker();
  mobileView();
  scrollMobilePanel(".scenes-panel");
}

function revisionDraftKey(revision) {
  return `${state.selectedJob}:${state.selectedScene}:${revision}`;
}

function selectedRevision() {
  const revision = Number($("#revision-select").value);
  return state.sceneData?.revisions.find((item) => item.revision === revision) || null;
}

function revisionParameters(revision) {
  return revision?.parameters || state.sceneData.parameters;
}

function revisionDraft(activeDraft, revision) {
  if (!revision) return null;
  return state.revisionDrafts.get(revisionDraftKey(revision.revision))
    || (activeDraft?.source_revision === revision.revision ? activeDraft : null);
}

function renderSceneHeading(revision) {
  $("#scene-heading").textContent = state.working.title;
  const version = revision ? ` · Version ${revision.revision}` : "";
  $("#scene-kicker").textContent = `Scene ${state.selectedScene} · ${state.sceneData.record.state}${version}`;
  renderContinuationStatus(revision?.chunk_progress || null);
}

function renderContinuationStatus(progress) {
  const element = $("#continuation-status");
  renderChunkLineage(progress);
  if (!progress) {
    element.classList.add("hidden");
    element.textContent = "";
    return;
  }
  const pieces = [
    `Chunk ${progress.current_chunk}/${progress.total_chunks}`,
    humanize(progress.phase),
    `${progress.completed_chunks} complete`,
  ];
  if (progress.resumed) pieces.push("resumed");
  if (progress.failed_attempts) pieces.push(`${progress.failed_attempts} failed attempt${progress.failed_attempts === 1 ? "" : "s"}`);
  element.textContent = pieces.join(" · ");
  element.classList.remove("hidden");
}

function lineageValue(value) {
  if (value == null || value === "") return "None";
  return displayValue(value);
}

function lineageFact(label, value, options = {}) {
  if (value === undefined || value === null) return "";
  const classes = [
    "lineage-fact",
    options.wide ? "lineage-wide" : "",
    options.long ? "lineage-long" : "",
  ].filter(Boolean).join(" ");
  const text = escapeHtml(lineageValue(value));
  const rendered = options.code ? `<code>${text}</code>` : text;
  return `<div class="${classes}"><dt>${escapeHtml(label)}</dt><dd>${rendered}</dd></div>`;
}

function lineageWindow(start, end) {
  if (start == null && end == null) return null;
  return `[${start ?? "?"}, ${end ?? "?"})`;
}

function lineageStateBadge(value) {
  const stateValue = value || "not_started";
  const stateClass = String(stateValue).toLowerCase().replace(/[^a-z0-9_-]/g, "_");
  return `<span class="badge ${stateClass}">${escapeHtml(humanize(stateValue))}</span>`;
}

function lineageLoras(loras) {
  if (!Array.isArray(loras) || !loras.length) return null;
  return loras.map((item) => {
    if (!item || typeof item !== "object") return lineageValue(item);
    const filename = item.filename || item.name || "Unnamed LoRA";
    return item.weight == null ? filename : `${filename} · weight ${item.weight}`;
  }).join("\n");
}

function lineageGroup(title, facts) {
  const content = facts.filter(Boolean).join("");
  if (!content) return "";
  return `
    <section class="lineage-group">
      <h5>${escapeHtml(title)}</h5>
      <dl class="chunk-lineage-grid">${content}</dl>
    </section>
  `;
}

function lineageProductionFacts(production) {
  if (!production || typeof production !== "object" || Array.isArray(production)) return [];
  return Object.entries(production).map(([key, value]) =>
    lineageFact(humanize(key), value)
  );
}

function renderChunkAttempt(attempt, acceptedAttemptNumber, chunkIndex, openAttempts) {
  const attemptNumber = Number(attempt?.attempt_number || 0);
  const accepted = acceptedAttemptNumber != null
    && attemptNumber === Number(acceptedAttemptNumber);
  const attemptKey = `${chunkIndex}:${attemptNumber}`;
  const open = openAttempts.has(attemptKey) ? " open" : "";
  const error = attempt?.error
    ? `<div class="lineage-error"><strong>Attempt error</strong><span>${escapeHtml(attempt.error)}</span></div>`
    : "";
  const mandatoryLoras = lineageLoras(attempt?.mandatory_loras);

  return `
    <details class="chunk-attempt-card${accepted ? " accepted" : ""}" data-attempt-key="${attribute(attemptKey)}"${open}>
      <summary class="chunk-attempt-summary">
        <span>Attempt ${attemptNumber || "?"}</span>
        ${lineageStateBadge(attempt?.state)}
        ${accepted ? '<span class="accepted-chip">Accepted attempt</span>' : ""}
      </summary>
      <div class="chunk-attempt-body">
        ${error}
        ${lineageGroup("Attempt", [
          lineageFact("Seed", attempt?.seed, {code: true}),
          lineageFact("Variation index", attempt?.variation_index),
          lineageFact("Stage 1 prompt ID", attempt?.stage1_prompt_id, {code: true}),
          lineageFact("Stage 2 prompt ID", attempt?.stage2_prompt_id, {code: true}),
          lineageFact("Artifact manifest", attempt?.artifact_manifest_path, {code: true, wide: true}),
          lineageFact("Rendered video", attempt?.video_path, {code: true, wide: true}),
        ])}
        ${lineageGroup("Workflow hashes", [
          lineageFact("Stage 1 workflow", attempt?.stage1_workflow_sha256, {code: true, wide: true}),
          lineageFact("Stage 2 workflow", attempt?.stage2_workflow_sha256, {code: true, wide: true}),
        ])}
        ${lineageGroup("Artifact hashes", [
          lineageFact("Upstream artifact", attempt?.upstream_artifact_hash, {code: true, wide: true}),
          lineageFact("Output artifact", attempt?.artifact_hash, {code: true, wide: true}),
          lineageFact("Raw video", attempt?.raw_video_sha256, {code: true, wide: true}),
        ])}
        ${lineageGroup("Model hashes", [
          lineageFact("Stage 1 checkpoint", attempt?.stage1_checkpoint_sha256, {code: true, wide: true}),
          lineageFact("Stage 2 checkpoint", attempt?.stage2_checkpoint_sha256, {code: true, wide: true}),
        ])}
        ${lineageGroup("Model files", [
          lineageFact("Checkpoint", attempt?.checkpoint_filename, {code: true}),
          lineageFact("Text encoder", attempt?.text_encoder_filename, {code: true}),
          lineageFact("Spatial upscaler", attempt?.spatial_upscaler_filename, {code: true}),
          lineageFact("Mandatory LoRAs", mandatoryLoras, {code: true, wide: true}),
        ])}
        ${lineageGroup("Node contracts", [
          lineageFact("Node-contract hash", attempt?.node_contracts_sha256, {code: true, wide: true}),
          lineageFact("Implementation hash", attempt?.implementation_sha256, {code: true, wide: true}),
        ])}
        ${lineageGroup("Production profile", lineageProductionFacts(attempt?.production))}
      </div>
    </details>
  `;
}

function renderChunkLineage(progress) {
  const element = $("#chunk-lineage");
  const chunks = Array.isArray(progress?.chunks) ? progress.chunks : [];
  if (!chunks.length) {
    element.classList.add("hidden");
    element.innerHTML = "";
    delete element.dataset.lineageKey;
    return;
  }

  const lineageKey = `${progress.job_id || ""}:${progress.scene_id || ""}:${progress.revision || ""}`;
  const sameLineage = element.dataset.lineageKey === lineageKey;
  const openChunks = new Set(
    sameLineage
      ? [...element.querySelectorAll("details.chunk-lineage-card[open]")]
        .map((item) => item.dataset.chunkIndex)
      : [],
  );
  const openAttempts = new Set(
    sameLineage
      ? [...element.querySelectorAll("details.chunk-attempt-card[open]")]
        .map((item) => item.dataset.attemptKey)
      : [],
  );
  const activeChunkIndex = Math.max(0, Number(progress.current_chunk || 1) - 1);

  const cards = chunks.map((chunk) => {
    const chunkIndex = Number(chunk.chunk_index ?? (chunk.chunk_number || 1) - 1);
    const chunkNumber = Number(chunk.chunk_number || chunkIndex + 1);
    const chunkKey = String(chunkIndex);
    const open = (
      openChunks.has(chunkKey)
      || (!sameLineage && chunkIndex === activeChunkIndex)
    ) ? " open" : "";
    const globalWindow = lineageWindow(
      chunk.global_window_start_frame,
      chunk.global_window_end_frame_exclusive,
    );
    const newWindow = lineageWindow(
      chunk.global_new_start_frame,
      chunk.global_new_end_frame_exclusive,
    );
    const segmentIndices = Array.isArray(chunk.segment_indices)
      ? (
        chunk.segment_indices.length
          ? chunk.segment_indices.map((index) => `Beat ${Number(index) + 1}`).join(", ")
          : "None"
      )
      : null;
    const attempts = Array.isArray(chunk.attempts) ? chunk.attempts : [];
    const attemptCards = attempts.map((attempt) =>
      renderChunkAttempt(
        attempt,
        chunk.accepted_attempt_number,
        chunkIndex,
        openAttempts,
      )
    ).join("");
    const chunkError = chunk.error
      ? `<div class="lineage-error"><strong>Chunk error</strong><span>${escapeHtml(chunk.error)}</span></div>`
      : "";

    return `
      <details class="chunk-lineage-card" data-chunk-index="${attribute(chunkKey)}"${open}>
        <summary class="chunk-lineage-summary">
          <span class="chunk-lineage-name">Chunk ${chunkNumber}</span>
          ${lineageStateBadge(chunk.state)}
          <span class="chunk-lineage-window">${escapeHtml(globalWindow || `${chunk.model_window_frames ?? "?"} model frames`)}</span>
          ${chunk.accepted_attempt_number == null ? "" : `<span class="accepted-chip">Accepted #${escapeHtml(chunk.accepted_attempt_number)}</span>`}
        </summary>
        <div class="chunk-lineage-body">
          ${chunkError}
          <dl class="chunk-lineage-grid">
            ${lineageFact("Resolved prompt", chunk.prompt, {wide: true, long: true})}
            ${lineageFact("Resolved negative", chunk.negative, {wide: true, long: true})}
            ${lineageFact("Seed", chunk.seed, {code: true})}
            ${lineageFact("Variation index", chunk.variation_index)}
            ${lineageFact("Model window frames", chunk.model_window_frames)}
            ${lineageFact("New transition frames", chunk.new_transition_frames)}
            ${lineageFact("Global model window", globalWindow, {code: true})}
            ${lineageFact("Global new-frame window", newWindow, {code: true})}
            ${lineageFact("Resolved segments", segmentIndices, {wide: true})}
            ${lineageFact("Prompt segmentation quality", chunk.prompt_segmentation_quality, {wide: true})}
            ${lineageFact("Accepted attempt", chunk.accepted_attempt_number)}
            ${lineageFact("Accepted artifact hash", chunk.accepted_artifact_hash, {code: true, wide: true})}
          </dl>
          <section class="chunk-attempts" aria-label="Chunk ${chunkNumber} generation attempts">
            <h4>Generation attempts</h4>
            ${attemptCards || '<p class="muted">No generation attempts recorded yet.</p>'}
          </section>
        </div>
      </details>
    `;
  }).join("");

  element.dataset.lineageKey = lineageKey;
  element.innerHTML = `
    <div class="chunk-lineage-header">
      <div>
        <p class="eyebrow">Audit trail</p>
        <h3>Continuation lineage</h3>
      </div>
      <span class="muted">${chunks.length} chunk${chunks.length === 1 ? "" : "s"}</span>
    </div>
    <div class="chunk-lineage-list">${cards}</div>
  `;
  element.classList.remove("hidden");
}

function renderRevisionPicker(preferredRevision = null) {
  const select = $("#revision-select");
  select.innerHTML = state.sceneData.revisions.map((revision) =>
    `<option value="${revision.revision}">Version ${revision.revision} · ${humanize(revision.state)}${revision.revision === 1 ? " · original" : ""}</option>`
  ).join("");
  const requested = Number(preferredRevision);
  const available = state.sceneData.revisions.some((item) => item.revision === requested);
  select.value = available ? requested : state.sceneData.revisions[0]?.revision || "";
  renderMedia();
}

function renderMedia() {
  const revision = selectedRevision();
  const image = $("#frame-preview");
  const video = $("#video-preview");
  image.src = revision?.frame_url || "";
  image.style.visibility = revision?.frame_url ? "visible" : "hidden";
  video.src = revision?.video_url || "";
  video.style.visibility = revision?.video_url ? "visible" : "hidden";
  video.load();
}

function selectRevisionParameters() {
  const key = `${state.selectedJob}:${state.selectedScene}`;
  const revision = selectedRevision();
  if (!revision) return;
  const saved = revisionDraft(state.drafts.get(key), revision);
  state.working = normalizeSceneParameters(
    saved?.parameters || revisionParameters(revision),
  );
  renderSceneHeading(revision);
  renderMedia();
  renderForm();
  if ($("#mark-remake").checked) saveDraft();
  setEditable($("#mark-remake").checked);
}

function renderForm() {
  syncProductionFrameProfile();
  $$("[data-path]").forEach((input) => {
    const value = getPath(state.working, input.dataset.path);
    input.value = value ?? "";
  });
  renderContext("#scene-context", state.working.scene_context);
  renderContext("#job-context", state.working.job_context);
  renderLoraEditor("#character-lora", "Global character LoRA", [state.working.character.global_lora], "character", false);
  renderOptionalLtxCharacterLora();
  renderLoraEditor("#t2i-loras", "Scene T2I LoRAs", state.working.t2i.loras, "t2i", true);
  renderLoraEditor("#i2v-loras", "Scene I2V LoRAs", state.working.i2v.loras, "i2v", true);
  $("#mandatory-loras").innerHTML = `<span class="muted">Mandatory LTX LoRAs</span>` +
    state.working.i2v.mandatory_loras.map((item) => `<span class="locked-chip">🔒 ${escapeHtml(item.name)} · ${item.weight}</span>`).join("");
  renderPasses();
  renderFaceDetailer();
  renderAdvanced();
  renderTemporalContinuation();
  renderProductionProfile();
  bindInputs();
}

function continuationFrameCounts(seconds, fps) {
  const numericSeconds = Number(seconds);
  const numericFps = Number(fps);
  if (
    !Number.isFinite(numericSeconds)
    || numericSeconds <= 0
    || !Number.isFinite(numericFps)
    || numericFps <= 0
  ) {
    return null;
  }
  const rawFrames = numericSeconds * numericFps;
  const lowerFrame = Math.floor(rawFrames);
  const fraction = rawFrames - lowerFrame;
  const roundedFrames = fraction === 0.5
    ? (lowerFrame % 2 === 0 ? lowerFrame : lowerFrame + 1)
    : Math.round(rawFrames);
  const timelineFrames = Math.max(1, roundedFrames);
  const generationMasterFrames = Math.max(
    9,
    8 * Math.ceil(Math.max(timelineFrames - 1, 0) / 8) + 1,
  );
  const legacyFrameCount = Math.max(
    9,
    8 * Math.ceil(rawFrames / 8) + 1,
  );
  return { timelineFrames, generationMasterFrames, legacyFrameCount };
}

function syncProductionFrameProfile() {
  const profile = state.working.production_profile;
  const counts = continuationFrameCounts(
    state.working.estimated_seconds,
    profile.fps,
  );
  if (!counts) return;
  profile.timeline_output_frames = counts.timelineFrames;
  profile.generation_master_frames = counts.generationMasterFrames;
  profile.frame_count = counts.legacyFrameCount;
}

function renderProductionProfile() {
  const profile = state.working.production_profile;
  const route = resolvedContinuationRoute();
  const frameItems = route.continuation
    ? [
        `${profile.timeline_output_frames} exact final timeline frames`,
        `${profile.generation_master_frames} LTX generation-master frames`,
        "8n + 1 generation master",
      ]
    : [
        `${profile.frame_count} frames`,
        "8n + 1",
      ];
  $("#production-profile").innerHTML = [
    `${profile.width} × ${profile.height}`,
    `${profile.fps} fps`,
    route.label,
    ...frameItems,
    "32 sec maximum",
  ].map((item) => `<span class="profile-item">🔒 ${item}</span>`).join("");
  const finalFrames = $("#continuation-final-frames");
  const masterFrames = $("#continuation-master-frames");
  if (finalFrames) {
    finalFrames.textContent =
      `🔒 ${profile.timeline_output_frames} final output frames`;
  }
  if (masterFrames) {
    masterFrames.textContent =
      `🔒 ${profile.generation_master_frames} generation-master frames`;
  }
}

function resolvedContinuationRoute() {
  const profile = state.working.production_profile;
  if (profile.timeline_output_frames <= 121) {
    return {
      continuation: false,
      label: "Resolved route: legacy single window (121 frames or fewer)",
    };
  }
  const continuation = state.working.i2v.temporal_continuation;
  const explicit = continuation?.enabled;
  const rolloutMode = state.status?.continuation_mode || "explicit";
  const enabled = rolloutMode !== "disabled"
    && explicit !== false
    && (rolloutMode === "auto" || explicit === true);
  return {
    continuation: enabled,
    label: enabled
      ? "Resolved route: chunked continuation"
      : "Resolved route: legacy single window",
  };
}

function renderContext(selector, data) {
  $(selector).innerHTML = Object.entries(data || {}).map(([key, value]) => `
    <dl class="context-item"><dt>${humanize(key)}</dt><dd>${escapeHtml(displayValue(value))}</dd></dl>
  `).join("") || `<p class="muted">No additional context.</p>`;
}

function renderLoraEditor(selector, title, loras, stage, allowMany) {
  const root = $(selector);
  const localLoras = state.options[
    stage === "i2v" || stage === "ltx_character" ? "i2v_loras" : "t2i_loras"
  ] || [];
  root.innerHTML = `
    <div class="lora-heading"><h4>${title}</h4>${allowMany ? `<button type="button" class="secondary add-lora" data-stage="${stage}">+ Add LoRA</button>` : ""}</div>
    <div class="lora-list">${loras.map((lora, index) => `
      <div class="lora-card">
        <div class="lora-heading"><strong>${escapeHtml(lora.name || "New LoRA")}</strong>${allowMany ? `<button type="button" class="remove-lora" data-stage="${stage}" data-index="${index}">Remove</button>` : ""}</div>
        <div class="field-grid">
          <label class="local-lora-field">Installed local file<select data-lora-picker="true" data-lora-stage="${stage}" data-lora-index="${index}">${localLoraOptions(localLoras, lora.name)}</select></label>
          <label>Name<input data-lora-stage="${stage}" data-lora-index="${index}" data-lora-key="name" value="${attribute(lora.name)}"></label>
          <label>Download URL<input data-lora-stage="${stage}" data-lora-index="${index}" data-lora-key="download_url" value="${attribute(lora.download_url)}"></label>
          <label>Weight<input type="number" step="0.05" min="-4" max="4" data-lora-stage="${stage}" data-lora-index="${index}" data-lora-key="weight" value="${attribute(lora.weight)}"></label>
        </div>
        <p class="local-lora-note">Picker fills Name from live ComfyUI files. Keep valid URL; I2V still verifies LTX 2.x compatibility.</p>
      </div>`).join("")}</div>`;
  root.querySelector(".add-lora")?.addEventListener("click", () => {
    state.working[stage].loras.push({ name: "New LoRA", download_url: "https://civitai.com/api/download/models/", weight: 1 });
    renderForm();
    setEditable(true);
  });
  root.querySelectorAll(".remove-lora").forEach((button) => button.addEventListener("click", () => {
    state.working[button.dataset.stage].loras.splice(Number(button.dataset.index), 1);
    renderForm();
    setEditable(true);
  }));
}

function renderOptionalLtxCharacterLora() {
  const root = $("#ltx-character-lora");
  const lora = state.working.character.ltx_character_lora;
  if (!lora) {
    root.innerHTML = `
      <div class="lora-heading">
        <h4>LTX character LoRA (video)</h4>
        <button id="add-ltx-character-lora" type="button" class="secondary">+ Add video character LoRA</button>
      </div>
      <p class="local-lora-note">Optional LTX 2.x character adapter used only by video generation.</p>
    `;
    $("#add-ltx-character-lora").addEventListener("click", () => {
      state.working.character.ltx_character_lora = {
        name: "New LTX character LoRA",
        download_url: "https://civitai.com/api/download/models/",
        weight: 0.8,
      };
      renderForm();
      setEditable(true);
      saveDraft();
    });
    return;
  }
  renderLoraEditor(
    "#ltx-character-lora",
    "LTX character LoRA (video)",
    [lora],
    "ltx_character",
    false,
  );
  const heading = root.querySelector(".lora-heading");
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "remove-lora";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => {
    state.working.character.ltx_character_lora = null;
    renderForm();
    setEditable(true);
    saveDraft();
  });
  heading.append(remove);
}

function loraTarget(stage, index) {
  if (stage === "character") return state.working.character.global_lora;
  if (stage === "ltx_character") {
    return state.working.character.ltx_character_lora;
  }
  return state.working[stage].loras[index];
}

function localLoraName(value) {
  return String(value || "").split(/[\\/]/).pop();
}

function localLoraOptions(loras, currentName) {
  const selected = loras.find((item) => localLoraName(item).toLowerCase() === String(currentName || "").toLowerCase()) || "";
  return [`<option value="">Choose installed file…</option>`, ...loras.map((item) =>
    `<option value="${attribute(item)}" ${item === selected ? "selected" : ""}>${escapeHtml(item)}</option>`
  )].join("");
}

function renderPasses() {
  $("#t2i-passes").innerHTML = state.working.t2i.passes.map((pass, index) =>
    passCard(`T2I pass ${index + 1}`, `t2i.passes.${index}`, pass, false)
  ).join("");
  $("#i2v-passes").innerHTML = [
    passCard("I2V first pass", "i2v.first_pass", state.working.i2v.first_pass, true),
    passCard("I2V spatial-upscale pass", "i2v.second_pass", state.working.i2v.second_pass, true),
  ].join("");
}

function passCard(title, path, pass, isI2v) {
  const common = `
    ${selectField("Sampler", `${path}.sampler`, pass.sampler, state.options.samplers)}
    ${!isI2v ? selectField("Scheduler", `${path}.scheduler`, pass.scheduler, state.options.schedulers) : ""}
    ${numberField("CFG", `${path}.cfg`, pass.cfg, .1)}
    ${!isI2v ? numberField("Steps", `${path}.steps`, pass.steps, 1) : ""}
  `;
  const extra = isI2v ? `
    <label>Sigmas<input data-path="${path}.sigmas" data-sigmas="true" value="${attribute(pass.sigmas.join(", "))}"></label>
    ${numberField("Reference strength", `${path}.reference_strength`, pass.reference_strength, .05)}
    ${numberField("Image strength", `${path}.image_strength`, pass.image_strength, .05)}
    ${numberField("Image compression", `${path}.image_compression`, pass.image_compression, 1)}
  ` : `
    ${pass.denoise != null ? numberField("Denoise", `${path}.denoise`, pass.denoise, .01) : ""}
    ${pass.start_step != null ? numberField("Start step", `${path}.start_step`, pass.start_step, 1) : ""}
    ${pass.end_step != null ? numberField("End step", `${path}.end_step`, pass.end_step, 1) : ""}
  `;
  return `<div class="pass-card"><h4>${title}</h4><div class="field-grid two">${common}${extra}</div></div>`;
}

function selectField(label, path, value, options) {
  const values = options.includes(value) ? options : [value, ...options];
  return `<label>${label}<select data-path="${path}">${values.map((item) => `<option ${item === value ? "selected" : ""}>${escapeHtml(item)}</option>`).join("")}</select></label>`;
}

function numberField(label, path, value, step) {
  return `<label>${label}<input type="number" data-path="${path}" value="${attribute(value)}" step="${step}"></label>`;
}

function renderFaceDetailer() {
  const value = state.working.t2i.face_detailer;
  $("#face-detailer-wrap").classList.toggle("hidden", !value);
  if (!value) return;
  $("#face-detailer").innerHTML = Object.entries(value).filter(([key]) => key !== "enabled").map(([key, item]) => {
    if (key === "sampler") return selectField(humanize(key), `t2i.face_detailer.${key}`, item, state.options.samplers);
    if (key === "scheduler") return selectField(humanize(key), `t2i.face_detailer.${key}`, item, state.options.schedulers);
    if (typeof item === "number") return numberField(humanize(key), `t2i.face_detailer.${key}`, item, Number.isInteger(item) ? 1 : .01);
    return `<label>${humanize(key)}<input data-path="t2i.face_detailer.${key}" value="${attribute(item)}"></label>`;
  }).join("");
}

function renderAdvanced() {
  const entries = [
    ["Chunks", "i2v.chunking.chunks", state.working.i2v.chunking.chunks],
    ["Dimension threshold", "i2v.chunking.dimension_threshold", state.working.i2v.chunking.dimension_threshold],
    ["Upscaler model", "i2v.spatial_upscaler.model", state.working.i2v.spatial_upscaler.model],
    ["Tile size", "i2v.spatial_upscaler.tile_size", state.working.i2v.spatial_upscaler.tile_size],
    ["Overlap", "i2v.spatial_upscaler.overlap", state.working.i2v.spatial_upscaler.overlap],
    ["No-tile maximum", "i2v.spatial_upscaler.max_size_without_tiling", state.working.i2v.spatial_upscaler.max_size_without_tiling],
  ];
  $("#i2v-advanced").innerHTML = entries.map(([label, path, value]) =>
    typeof value === "number" ? numberField(label, path, value, 1) : `<label>${label}<input data-path="${path}" value="${attribute(value)}"></label>`
  ).join("");
}

function continuationRoutingValue() {
  const value = state.working.i2v.temporal_continuation;
  if (value == null) return "default";
  return value.enabled === false ? "disabled" : "enabled";
}

function ensureContinuationSettings(enabled = true) {
  const current = state.working.i2v.temporal_continuation;
  state.working.i2v.temporal_continuation = {
    ...(current || {}),
    enabled,
    strategy: "ltx23_latent_overlap_v1",
    fps: 24,
    base_window_transition_frames: 120,
    overlap_transition_frames: 24,
    seed_policy: "derived_v1",
  };
  delete state.working.i2v.temporal_continuation.requested_duration_seconds;
}

function anchorLines(value) {
  if (Array.isArray(value)) return value.join("\n");
  return value == null ? "" : String(value);
}

function renderTemporalContinuation() {
  const root = $("#temporal-continuation");
  if (!Array.isArray(state.working.i2v.segments)) {
    state.working.i2v.segments = [];
  }
  const continuity = state.working.i2v.continuity || {};
  const segments = state.working.i2v.segments;
  const profile = state.working.production_profile;
  root.innerHTML = `
    <div class="continuation-routing">
      <label>Long-scene routing
        <select id="continuation-routing">
          <option value="default" ${continuationRoutingValue() === "default" ? "selected" : ""}>Pipeline default</option>
          <option value="enabled" ${continuationRoutingValue() === "enabled" ? "selected" : ""}>Force chunked continuation</option>
          <option value="disabled" ${continuationRoutingValue() === "disabled" ? "selected" : ""}>Force legacy single window</option>
        </select>
      </label>
      <div class="locked-list continuation-locks">
        <span class="locked-chip">🔒 24 fps</span>
        <span class="locked-chip">🔒 120 base transitions</span>
        <span class="locked-chip">🔒 24-frame overlap</span>
        <span class="locked-chip">🔒 up to 96 new transitions</span>
        <span id="continuation-final-frames" class="locked-chip">🔒 ${profile.timeline_output_frames} final output frames</span>
        <span id="continuation-master-frames" class="locked-chip">🔒 ${profile.generation_master_frames} generation-master frames</span>
      </div>
    </div>
    <h4>Continuity anchors</h4>
    <div class="field-grid two continuity-grid">
      <label>Identity anchors<textarea rows="3" data-continuity-key="identity_anchors" data-continuity-list="true">${escapeHtml(anchorLines(continuity.identity_anchors))}</textarea></label>
      <label>Wardrobe anchors<textarea rows="3" data-continuity-key="wardrobe_anchors" data-continuity-list="true">${escapeHtml(anchorLines(continuity.wardrobe_anchors))}</textarea></label>
      <label>Environment anchors<textarea rows="3" data-continuity-key="environment_anchors" data-continuity-list="true">${escapeHtml(anchorLines(continuity.environment_anchors))}</textarea></label>
      <label>Camera axis<input data-continuity-key="camera_axis" value="${attribute(continuity.camera_axis || "")}"></label>
      <label>Screen direction<input data-continuity-key="screen_direction" value="${attribute(continuity.screen_direction || "")}"></label>
    </div>
    <div class="lora-heading continuation-segment-heading">
      <h4>Ordered scene beats</h4>
      <button id="add-continuation-segment" type="button" class="secondary">+ Add beat</button>
    </div>
    <p class="muted">Beats are mapped onto overlapping LTX windows. Large seed overrides stay as exact decimal text.</p>
    <p id="beat-coverage" class="muted">${escapeHtml(beatCoverageText())}</p>
    <div class="continuation-segments">
      ${segments.map((segment, index) => continuationSegmentCard(segment, index)).join("") || `<p class="muted">No explicit beats. The scene motion prompt will be reused with continuity instructions.</p>`}
    </div>
  `;
  $("#continuation-routing").addEventListener("change", (event) => {
    if (event.target.value === "default") {
      state.working.i2v.temporal_continuation = null;
    } else {
      ensureContinuationSettings(event.target.value === "enabled");
    }
    renderProductionProfile();
    saveDraft();
  });
  $$("[data-continuity-key]").forEach((input) => input.addEventListener("input", () => {
    const key = input.dataset.continuityKey;
    const value = input.dataset.continuityList
      ? input.value.split("\n").map((item) => item.trim()).filter(Boolean)
      : input.value;
    state.working.i2v.continuity = {
      ...(state.working.i2v.continuity || {}),
      [key]: value,
    };
    saveDraft();
  }));
  $$("[data-segment-key]").forEach((input) => input.addEventListener("input", () => {
    const segment = state.working.i2v.segments[Number(input.dataset.segmentIndex)];
    const key = input.dataset.segmentKey;
    if (input.dataset.segmentList) {
      segment[key] = input.value.split("\n").map((item) => item.trim()).filter(Boolean);
    } else if (input.dataset.segmentInteger || input.dataset.segmentNumber) {
      if (input.value === "") delete segment[key];
      else segment[key] = Number(input.value);
      if (key === "requested_duration_seconds") {
        delete segment.new_transition_frames;
      } else if (key === "new_transition_frames") {
        delete segment.requested_duration_seconds;
      }
    } else if (key === "seed_override") {
      if (input.value.trim()) segment[key] = input.value.trim();
      else delete segment[key];
    } else {
      segment[key] = input.value;
    }
    updateBeatCoverage();
    saveDraft();
  }));
  $$("[data-segment-timing-mode]").forEach((select) => select.addEventListener("change", () => {
    const segment = state.working.i2v.segments[Number(select.dataset.segmentIndex)];
    if (select.value === "transitions") {
      delete segment.requested_duration_seconds;
      if (segment.new_transition_frames == null) segment.new_transition_frames = 96;
    } else {
      delete segment.new_transition_frames;
      if (segment.requested_duration_seconds == null) {
        segment.requested_duration_seconds = 4;
      }
    }
    renderForm();
    setEditable(true);
    saveDraft();
  }));
  $$(".remove-continuation-segment").forEach((button) => button.addEventListener("click", () => {
    state.working.i2v.segments.splice(Number(button.dataset.segmentIndex), 1);
    state.working.i2v.segments.forEach((segment, index) => { segment.index = index; });
    renderForm();
    setEditable(true);
    saveDraft();
  }));
  $("#add-continuation-segment").addEventListener("click", () => {
    const index = state.working.i2v.segments.length;
    state.working.i2v.segments.push({
      index,
      requested_duration_seconds: 4,
      positive_prompt: state.working.i2v.prompt,
      negative_prompt_additions: [],
      variation_index: 0,
    });
    renderForm();
    setEditable(true);
    saveDraft();
  });
}

function segmentTimingMode(segment) {
  return segment.new_transition_frames != null ? "transitions" : "duration";
}

function beatCoverageText() {
  const segments = state.working.i2v.segments;
  if (!segments.length) return "Beat coverage: no explicit beats.";
  const fps = Number(state.working.production_profile.fps);
  const target = Number(state.working.production_profile.timeline_output_frames);
  const covered = segments.reduce((total, segment) => {
    if (segment.new_transition_frames != null) {
      return total + Math.max(1, Number(segment.new_transition_frames) || 0);
    }
    return total + (
      continuationFrameCounts(segment.requested_duration_seconds, fps)?.timelineFrames
      || 0
    );
  }, 0);
  const difference = covered - target;
  const status = difference === 0
    ? "complete"
    : difference < 0
      ? `${Math.abs(difference)} frame(s) short`
      : `${difference} frame(s) over`;
  return `Beat coverage: ${covered}/${target} timeline frames · ${status}.`;
}

function updateBeatCoverage() {
  const element = $("#beat-coverage");
  if (element) element.textContent = beatCoverageText();
}

function continuationSegmentCard(segment, index) {
  const timingMode = segmentTimingMode(segment);
  const timingInput = timingMode === "transitions"
    ? `<label>New transitions (8n)<input type="number" min="0" step="8" data-segment-index="${index}" data-segment-key="new_transition_frames" data-segment-integer="true" value="${attribute(segment.new_transition_frames ?? "")}"></label>`
    : `<label>Requested seconds<input type="number" min="0.04" max="32" step="0.01" data-segment-index="${index}" data-segment-key="requested_duration_seconds" data-segment-number="true" value="${attribute(segment.requested_duration_seconds ?? "")}"></label>`;
  return `
    <div class="continuation-segment-card">
      <div class="lora-heading">
        <strong>Beat ${index + 1}</strong>
        <button type="button" class="remove-continuation-segment" data-segment-index="${index}">Remove</button>
      </div>
      <label>Positive prompt<textarea rows="4" data-segment-index="${index}" data-segment-key="positive_prompt">${escapeHtml(segment.positive_prompt || "")}</textarea></label>
      <div class="field-grid three segment-timing-grid">
        <label>Timing mode<select data-segment-timing-mode="true" data-segment-index="${index}">
          <option value="duration" ${timingMode === "duration" ? "selected" : ""}>Duration in seconds</option>
          <option value="transitions" ${timingMode === "transitions" ? "selected" : ""}>Exact transition count</option>
        </select></label>
        ${timingInput}
        <label>Variation index<input type="number" min="0" step="1" data-segment-index="${index}" data-segment-key="variation_index" data-segment-integer="true" value="${attribute(segment.variation_index ?? 0)}"></label>
      </div>
      <label>Negative additions<textarea rows="2" data-segment-index="${index}" data-segment-key="negative_prompt_additions" data-segment-list="true">${escapeHtml((segment.negative_prompt_additions || []).join("\n"))}</textarea></label>
      <label>Seed override (optional)<input type="text" inputmode="numeric" data-segment-index="${index}" data-segment-key="seed_override" value="${attribute(segment.seed_override ?? "")}"></label>
    </div>
  `;
}

function bindInputs() {
  $$("[data-path]").forEach((input) => {
    input.addEventListener("input", () => {
      let value = input.value;
      if (input.dataset.sigmas) value = value.split(",").map((item) => Number(item.trim())).filter((item) => !Number.isNaN(item));
      else if (input.type === "number") value = Number(value);
      setPath(state.working, input.dataset.path, value);
      if (input.dataset.path === "estimated_seconds") {
        syncProductionFrameProfile();
        renderProductionProfile();
      }
      saveDraft();
    });
  });
  $$("[data-lora-key]").forEach((input) => {
    input.addEventListener("input", () => {
      const stage = input.dataset.loraStage;
      const target = loraTarget(stage, Number(input.dataset.loraIndex));
      target[input.dataset.loraKey] = input.type === "number" ? Number(input.value) : input.value;
      saveDraft();
    });
  });
  $$("[data-lora-picker]").forEach((picker) => {
    picker.addEventListener("change", () => {
      if (!picker.value) return;
      const stage = picker.dataset.loraStage;
      const target = loraTarget(stage, Number(picker.dataset.loraIndex));
      target.name = localLoraName(picker.value);
      saveDraft();
      renderForm();
      setEditable(true);
    });
  });
}

function setEditable(editable) {
  $("#parameter-form").querySelectorAll("input, textarea, select, button").forEach((element) => {
    element.disabled = !editable;
  });
  $("#revision-select").disabled = false;
  $("#remake-mode").classList.toggle("hidden", !editable);
  $("#draft-state").textContent = editable ? "Changes are saved in the remake tray." : "Enable remake to edit parameters.";
}

function saveDraft() {
  const key = `${state.selectedJob}:${state.selectedScene}`;
  if (!$("#mark-remake").checked) return;
  const mode = $('input[name="remake-mode"]:checked').value;
  const sourceRevision = selectedRevision()?.revision || 1;
  const draft = {
    job_id: state.selectedJob,
    scene_id: state.selectedScene,
    source_revision: sourceRevision,
    remake_mode: mode,
    parameters: clone(state.working),
  };
  state.drafts.set(key, draft);
  state.revisionDrafts.set(revisionDraftKey(sourceRevision), draft);
  updateDraftCount();
}

function updateDraftCount() {
  $("#draft-count").textContent = state.drafts.size;
  $("#submit-batch").disabled = state.drafts.size === 0;
}

async function submitDrafts() {
  if (!state.drafts.size) return;
  try {
    state.pendingBatch = await api("/api/remake-batches", {
      method: "POST",
      body: JSON.stringify({ items: [...state.drafts.values()] }),
    });
    if (state.pendingBatch.active_render) {
      $("#collision-modal").classList.remove("hidden");
    } else {
      await submitPendingBatch("after_current");
    }
  } catch (error) {
    toast(error.message, true);
  }
}

async function submitPendingBatch(policy) {
  if (!state.pendingBatch) return;
  try {
    await api(`/api/remake-batches/${state.pendingBatch.batch_id}/submit`, {
      method: "POST",
      body: JSON.stringify({ collision_policy: policy }),
    });
    toast(`Remake batch queued with ${state.drafts.size} scene edit(s).`);
    state.drafts.clear();
    state.revisionDrafts.clear();
    state.pendingBatch = null;
    $("#collision-modal").classList.add("hidden");
    updateDraftCount();
    if (state.selectedScene) await selectScene(state.selectedScene);
  } catch (error) {
    toast(error.message, true);
  }
}

async function setManualFinalInclusion() {
  const included = $("#include-in-manual-final").checked;
  try {
    const result = await api(
      `/api/jobs/${encodeURIComponent(state.selectedJob)}/scenes/${state.selectedScene}/manual-final-inclusion`,
      { method: "PUT", body: JSON.stringify({ included }) },
    );
    state.sceneData.record.include_in_manual_final = result.include_in_manual_final;
    const scene = state.scenes.find((item) => item.scene_id === state.selectedScene);
    if (scene) scene.include_in_manual_final = result.include_in_manual_final;
    renderScenes();
    renderManualFinalControls();
    toast(included ? "Scene included in the manual project final." : "Scene excluded from the manual project final.");
  } catch (error) {
    $("#include-in-manual-final").checked = !included;
    toast(error.message, true);
  }
}

async function queueManualFinal() {
  if (!state.selectedJob) return;
  try {
    state.manualFinal = await api(
      `/api/jobs/${encodeURIComponent(state.selectedJob)}/manual-final`,
      { method: "POST" },
    );
    renderManualFinalControls();
    toast("Manual project final queued. It waits for active project work, then concatenates the selected clips.");
  } catch (error) {
    toast(error.message, true);
  }
}

function chunkProgressKey(progress) {
  return progress
    ? `${progress.job_id}:${progress.scene_id}:${progress.revision}`
    : null;
}

function applyChunkProgressToCaches(progress) {
  if (!progress) return;
  if (progress.job_id === state.selectedJob) {
    const scene = state.scenes.find(
      (item) => item.scene_id === progress.scene_id,
    );
    if (scene) scene.chunk_progress = progress;
    renderScenes();
  }
  if (
    progress.job_id === state.selectedJob
    && progress.scene_id === state.selectedScene
  ) {
    const revision = state.sceneData?.revisions.find(
      (item) => item.revision === progress.revision,
    );
    if (revision) revision.chunk_progress = progress;
    if (selectedRevision()?.revision === progress.revision) {
      renderContinuationStatus(progress);
    }
  }
}

async function refreshChunkProgress(progress) {
  const key = chunkProgressKey(progress);
  if (!key || state.progressRefreshes.has(key)) return;
  state.progressRefreshes.add(key);
  try {
    const sceneData = await api(
      `/api/jobs/${encodeURIComponent(progress.job_id)}/scenes/${progress.scene_id}`,
    );
    const latestProgress = sceneData.revisions[0]?.chunk_progress || null;
    if (progress.job_id === state.selectedJob) {
      const scene = state.scenes.find(
        (item) => item.scene_id === progress.scene_id,
      );
      if (scene) {
        scene.state = sceneData.record.state;
        scene.chunk_progress = latestProgress;
      }
      renderScenes();
    }
    if (
      progress.job_id === state.selectedJob
      && progress.scene_id === state.selectedScene
      && state.sceneData
    ) {
      state.sceneData.record = sceneData.record;
      sceneData.revisions.forEach((freshRevision) => {
        const cachedRevision = state.sceneData.revisions.find(
          (item) => item.revision === freshRevision.revision,
        );
        if (cachedRevision) {
          cachedRevision.chunk_progress = freshRevision.chunk_progress;
          cachedRevision.state = freshRevision.state;
          cachedRevision.frame_url = freshRevision.frame_url;
          cachedRevision.video_url = freshRevision.video_url;
        }
      });
      renderContinuationStatus(selectedRevision()?.chunk_progress || null);
      renderMedia();
    }
  } catch (error) {
    console.warn("Could not refresh continuation progress.", error);
  } finally {
    state.progressRefreshes.delete(key);
  }
}

function updateStatus(status) {
  state.status = status;
  const element = $("#status");
  const intake = typeof status.hold_new_jobs_for_review === "boolean"
    ? (status.hold_new_jobs_for_review ? "review hold" : "auto-start")
    : "intake unknown";
  element.textContent = `${humanize(status.pipeline_state)}${status.job_id ? ` · ${status.job_id}` : ""} · ${intake} · Comfy ${status.comfyui_running}/${status.comfyui_pending}`;
  element.className = `status-pill ${!status.comfyui_healthy || status.pipeline_error ? "error" : status.active_render ? "busy" : "healthy"}`;
  $("#approve-job").classList.toggle("hidden", status.pipeline_state !== "awaiting_review" || status.job_id !== state.selectedJob);
  const canCancel = Boolean(status.can_cancel_current_project);
  const cancelButton = $("#cancel-current-project");
  cancelButton.classList.toggle("hidden", !canCancel);
  cancelButton.disabled = !canCancel || state.cancellingProject;
  if (status.manual_final?.job_id === state.selectedJob) {
    state.manualFinal = status.manual_final;
    renderManualFinalControls();
  }
  const previousProgress = state.activeChunkProgress;
  const nextProgress = status.chunk_progress || null;
  if (nextProgress) {
    applyChunkProgressToCaches(nextProgress);
  }
  if (
    previousProgress
    && chunkProgressKey(previousProgress) !== chunkProgressKey(nextProgress)
  ) {
    void refreshChunkProgress(previousProgress);
  }
  state.activeChunkProgress = nextProgress;
  if (state.working) renderProductionProfile();
  mobileView();
}

async function cancelCurrentProject() {
  const jobId = state.status?.job_id || "the current project";
  const confirmed = window.confirm(
    `Cancel ${jobId} and move on?\n\n` +
      "Unfinished scenes are marked cancelled. History and completed scenes stay saved for later remake. " +
      "The supervisor will check email for the next job, then send a request only if none is waiting.",
  );
  if (!confirmed) return;
  state.cancellingProject = true;
  updateStatus(state.status || {});
  try {
    const result = await api("/api/pipeline/cancel-current", { method: "POST" });
    toast(
      `Cancelled ${result.job_id}. Looking for the next job` +
        (result.cancelled_prompts?.length
          ? ` (stopped ${result.cancelled_prompts.length} ComfyUI prompt(s)).`
          : "."),
    );
    await loadJobs();
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.cancellingProject = false;
    if (state.status) updateStatus(state.status);
  }
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value ?? "";
  return element.innerHTML;
}

function attribute(value) {
  return escapeHtml(String(value ?? "")).replaceAll('"', "&quot;");
}

$("#refresh").addEventListener("click", loadJobs);
$("#back-to-projects").addEventListener("click", backToProjects);
$("#back-to-scenes").addEventListener("click", backToScenes);
$("#mobile-scene-select").addEventListener("change", (event) => {
  const sceneId = Number(event.target.value);
  if (Number.isInteger(sceneId) && sceneId > 0) selectScene(sceneId);
});
$("#revision-select").addEventListener("change", selectRevisionParameters);
$("#include-in-manual-final").addEventListener("change", setManualFinalInclusion);
$("#render-project-final").addEventListener("click", queueManualFinal);
$("#mark-remake").addEventListener("change", () => {
  const key = `${state.selectedJob}:${state.selectedScene}`;
  if ($("#mark-remake").checked) {
    saveDraft();
  } else {
    state.drafts.delete(key);
    for (const draftKey of state.revisionDrafts.keys()) {
      if (draftKey.startsWith(`${key}:`)) state.revisionDrafts.delete(draftKey);
    }
    state.working = normalizeSceneParameters(
      revisionParameters(selectedRevision()),
    );
    renderSceneHeading(selectedRevision());
    renderForm();
    updateDraftCount();
  }
  setEditable($("#mark-remake").checked);
});
$$('input[name="remake-mode"]').forEach((input) => input.addEventListener("change", saveDraft));
$("#submit-batch").addEventListener("click", submitDrafts);
$("#queue-after").addEventListener("click", () => submitPendingBatch("after_current"));
$("#interrupt-now").addEventListener("click", () => submitPendingBatch("interrupt_current"));
$("#cancel-modal").addEventListener("click", () => $("#collision-modal").classList.add("hidden"));
$("#approve-job").addEventListener("click", async () => {
  try {
    await api(`/api/jobs/${encodeURIComponent(state.selectedJob)}/approve`, { method: "POST" });
    toast("Job approved and queued.");
  } catch (error) {
    toast(error.message, true);
  }
});
$("#cancel-current-project").addEventListener("click", cancelCurrentProject);

async function init() {
  try {
    state.options = await api("/api/options");
  } catch (error) {
    toast(`Live sampler options unavailable: ${error.message}`, true);
  }
  await loadJobs();
  mobileView();
  const stream = new EventSource("/api/events");
  stream.onmessage = (event) => updateStatus(JSON.parse(event.data));
  stream.onerror = () => updateStatus({ pipeline_state: "disconnected", comfyui_healthy: false, comfyui_running: 0, comfyui_pending: 0 });
}

window.addEventListener("resize", mobileView);
mobileQuery.addEventListener?.("change", mobileView);

init().catch((error) => toast(error.message, true));
