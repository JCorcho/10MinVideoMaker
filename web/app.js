const state = {
  jobs: [],
  scenes: [],
  selectedJob: null,
  selectedScene: null,
  sceneData: null,
  working: null,
  drafts: new Map(),
  options: { samplers: [], schedulers: [], t2i_loras: [], i2v_loras: [] },
  pendingBatch: null,
  status: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const mobileQuery = window.matchMedia("(max-width: 760px)");

function mobileView() {
  if (!mobileQuery.matches) {
    document.body.removeAttribute("data-mobile-view");
    document.documentElement.style.removeProperty("--mobile-topbar-height");
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
  renderJobs();
  renderMobileScenePicker();
  mobileView();
  const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
  const summary = state.jobs.find((item) => item.job_id === jobId);
  $("#job-title").textContent = job.display_name || summary?.display_name || job.job_id;
  $("#job-meta").textContent = `${job.character.name} · ${job.character.series} · ${job.character.base_model}`;
  state.scenes = job.scenes;
  $("#scenes").innerHTML = state.scenes.map((scene) => `
    <button class="list-card" data-scene="${scene.scene_id}">
      <strong>${String(scene.scene_id).padStart(2, "0")} · ${escapeHtml(scene.title)}</strong>
      <div class="card-row">${badge(scene.state)}<span>${scene.revision_count} version${scene.revision_count === 1 ? "" : "s"}</span></div>
    </button>
  `).join("");
  $$("[data-scene]").forEach((button) => button.addEventListener("click", () => selectScene(Number(button.dataset.scene))));
  $("#approve-job").classList.toggle("hidden", state.status?.pipeline_state !== "awaiting_review" || state.status?.job_id !== jobId);
  renderMobileScenePicker();
  mobileView();
  scrollMobilePanel(".scenes-panel");
}

async function selectScene(sceneId) {
  state.selectedScene = sceneId;
  mobileView();
  state.sceneData = await api(`/api/jobs/${encodeURIComponent(state.selectedJob)}/scenes/${sceneId}`);
  const key = `${state.selectedJob}:${sceneId}`;
  state.working = clone(state.drafts.get(key)?.parameters || state.sceneData.parameters);
  $("#empty-state").classList.add("hidden");
  $("#scene-detail").classList.remove("hidden");
  $("#scene-heading").textContent = state.working.title;
  $("#scene-kicker").textContent = `Scene ${sceneId} · ${state.sceneData.record.state}`;
  $$("[data-scene]").forEach((button) => button.classList.toggle("selected", Number(button.dataset.scene) === sceneId));
  renderRevisionPicker();
  renderForm();
  const draft = state.drafts.get(key);
  $("#mark-remake").checked = Boolean(draft);
  $("#remake-mode").classList.toggle("hidden", !draft);
  if (draft) {
    $(`input[name="remake-mode"][value="${draft.remake_mode}"]`).checked = true;
  }
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
  $("#job-title").textContent = "Choose a project";
  $("#job-meta").textContent = "";
  $("#scenes").innerHTML = "";
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

function renderRevisionPicker() {
  const select = $("#revision-select");
  select.innerHTML = state.sceneData.revisions.map((revision) =>
    `<option value="${revision.revision}">Version ${revision.revision} · ${humanize(revision.state)}${revision.revision === 1 ? " · original" : ""}</option>`
  ).join("");
  select.value = state.sceneData.revisions[0]?.revision || "";
  renderMedia();
}

function renderMedia() {
  const revision = state.sceneData.revisions.find((item) => item.revision === Number($("#revision-select").value));
  const image = $("#frame-preview");
  const video = $("#video-preview");
  image.src = revision?.frame_url || "";
  image.style.visibility = revision?.frame_url ? "visible" : "hidden";
  video.src = revision?.video_url || "";
  video.style.visibility = revision?.video_url ? "visible" : "hidden";
  video.load();
}

function renderForm() {
  $$("[data-path]").forEach((input) => {
    const value = getPath(state.working, input.dataset.path);
    input.value = value ?? "";
  });
  renderContext("#scene-context", state.working.scene_context);
  renderContext("#job-context", state.working.job_context);
  renderLoraEditor("#character-lora", "Global character LoRA", [state.working.character.global_lora], "character", false);
  renderLoraEditor("#t2i-loras", "Scene T2I LoRAs", state.working.t2i.loras, "t2i", true);
  renderLoraEditor("#i2v-loras", "Scene I2V LoRAs", state.working.i2v.loras, "i2v", true);
  $("#mandatory-loras").innerHTML = `<span class="muted">Mandatory LTX LoRAs</span>` +
    state.working.i2v.mandatory_loras.map((item) => `<span class="locked-chip">🔒 ${escapeHtml(item.name)} · ${item.weight}</span>`).join("");
  renderPasses();
  renderFaceDetailer();
  renderAdvanced();
  const profile = state.working.production_profile;
  $("#production-profile").innerHTML = [
    `${profile.width} × ${profile.height}`,
    `${profile.fps} fps`,
    `${profile.frame_count} frames`,
    "8n + 1",
    "32 sec maximum",
  ].map((item) => `<span class="profile-item">🔒 ${item}</span>`).join("");
  bindInputs();
}

function renderContext(selector, data) {
  $(selector).innerHTML = Object.entries(data || {}).map(([key, value]) => `
    <dl class="context-item"><dt>${humanize(key)}</dt><dd>${escapeHtml(displayValue(value))}</dd></dl>
  `).join("") || `<p class="muted">No additional context.</p>`;
}

function renderLoraEditor(selector, title, loras, stage, allowMany) {
  const root = $(selector);
  const localLoras = state.options[stage === "i2v" ? "i2v_loras" : "t2i_loras"] || [];
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

function bindInputs() {
  $$("[data-path]").forEach((input) => {
    input.addEventListener("input", () => {
      let value = input.value;
      if (input.dataset.sigmas) value = value.split(",").map((item) => Number(item.trim())).filter((item) => !Number.isNaN(item));
      else if (input.type === "number") value = Number(value);
      setPath(state.working, input.dataset.path, value);
      saveDraft();
    });
  });
  $$("[data-lora-key]").forEach((input) => {
    input.addEventListener("input", () => {
      const stage = input.dataset.loraStage;
      const target = stage === "character"
        ? state.working.character.global_lora
        : state.working[stage].loras[Number(input.dataset.loraIndex)];
      target[input.dataset.loraKey] = input.type === "number" ? Number(input.value) : input.value;
      saveDraft();
    });
  });
  $$("[data-lora-picker]").forEach((picker) => {
    picker.addEventListener("change", () => {
      if (!picker.value) return;
      const stage = picker.dataset.loraStage;
      const target = stage === "character"
        ? state.working.character.global_lora
        : state.working[stage].loras[Number(picker.dataset.loraIndex)];
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
  state.drafts.set(key, {
    job_id: state.selectedJob,
    scene_id: state.selectedScene,
    remake_mode: mode,
    parameters: clone(state.working),
  });
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
    state.pendingBatch = null;
    $("#collision-modal").classList.add("hidden");
    updateDraftCount();
    if (state.selectedScene) await selectScene(state.selectedScene);
  } catch (error) {
    toast(error.message, true);
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
  mobileView();
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
$("#revision-select").addEventListener("change", renderMedia);
$("#mark-remake").addEventListener("change", () => {
  const key = `${state.selectedJob}:${state.selectedScene}`;
  if ($("#mark-remake").checked) {
    saveDraft();
  } else {
    state.drafts.delete(key);
    state.working = clone(state.sceneData.parameters);
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
