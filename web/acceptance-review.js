const reviewState = { runs: [], document: null, selectedRun: null, selectedCase: null };
const $ = (selector) => document.querySelector(selector);

async function request(path) {
  const response = await fetch(path);
  const body = response.headers.get("content-type")?.includes("application/json")
    ? await response.json()
    : null;
  if (!response.ok) throw new Error(body?.detail || `Request failed (${response.status})`);
  return body;
}

function setStatus(message, error = false) {
  const status = $("#review-status");
  status.textContent = message;
  status.classList.toggle("error", error);
}

function updateRunOptions() {
  const select = $("#run-select");
  select.innerHTML = reviewState.runs.map((run) =>
    `<option value="${run.run_id}">${run.run_id} · job ${run.source_job_id ?? "unknown"}</option>`
  ).join("");
  select.value = reviewState.selectedRun || "";
}

function updateCaseOptions() {
  const select = $("#case-select");
  const cases = reviewState.document.case_order;
  select.innerHTML = cases.map((name) => {
    const title = reviewState.document.cases[name].title;
    return `<option value="${name}">${title}</option>`;
  }).join("");
  select.value = reviewState.selectedCase;
}

function setVideoSource(video, url) {
  video.pause();
  if (video.src.endsWith(url)) return;
  video.src = url;
  video.load();
}

function renderStills(caseDocument) {
  $("#stills").innerHTML = caseDocument.stills.map((still) => `
    <figure class="still-card">
      <img src="${still.url}" alt="${still.label}" loading="lazy">
      <figcaption><strong>${still.side === "base" ? "Base" : "Continuation"}</strong>${still.label}</figcaption>
    </figure>
  `).join("");
}

function renderCase() {
  const caseDocument = reviewState.document.cases[reviewState.selectedCase];
  const boundary = caseDocument.boundary;
  setVideoSource($("#base-video"), reviewState.document.base.video_url);
  setVideoSource($("#case-video"), caseDocument.video_url);
  $("#case-title").textContent = caseDocument.title;
  $("#case-summary").textContent = caseDocument.summary;
  $("#base-label").textContent = `Compare base frames ${boundary.left[0]}–${boundary.left[1]}; final left frame is seam reference.`;
  $("#case-label").textContent = `Compare continuation frames ${boundary.right[0]}–${boundary.right[1]}; final right frame is first new motion.`;
  $("#boundary-summary").textContent = `Exact boundary: base ${boundary.left[1]} then continuation ${boundary.right[0]}. Use “Show exact seam” to seek both videos.`;
  $("#show-seam").disabled = false;
  renderStills(caseDocument);
  const url = new URL(window.location.href);
  url.searchParams.set("run", reviewState.selectedRun);
  window.history.replaceState({}, "", url);
}

function showSeam() {
  const caseDocument = reviewState.document.cases[reviewState.selectedCase];
  const fps = reviewState.document.fps;
  const base = $("#base-video");
  const continuation = $("#case-video");
  base.pause();
  continuation.pause();
  base.currentTime = caseDocument.boundary.left[1] / fps;
  continuation.currentTime = caseDocument.boundary.right[0] / fps;
  setStatus(`Showing base frame ${caseDocument.boundary.left[1]} and continuation frame ${caseDocument.boundary.right[0]}.`);
}

async function loadRun(runId) {
  setStatus("Loading selected review…");
  reviewState.document = await request(`/api/acceptance-runs/${encodeURIComponent(runId)}`);
  reviewState.selectedRun = runId;
  reviewState.selectedCase = reviewState.document.case_order.includes(reviewState.selectedCase)
    ? reviewState.selectedCase
    : reviewState.document.case_order[0];
  updateRunOptions();
  updateCaseOptions();
  renderCase();
  setStatus("Review media prepares on first play. Raw clips stay unchanged.");
}

async function initialize() {
  try {
    reviewState.runs = await request("/api/acceptance-runs");
    if (!reviewState.runs.length) throw new Error("No completed continuation acceptance run is available.");
    const requested = new URLSearchParams(window.location.search).get("run");
    reviewState.selectedRun = reviewState.runs.some((run) => run.run_id === requested)
      ? requested
      : reviewState.runs[0].run_id;
    updateRunOptions();
    await loadRun(reviewState.selectedRun);
  } catch (error) {
    setStatus(error.message || "Acceptance review failed to load.", true);
  }
}

$("#run-select").addEventListener("change", (event) => loadRun(event.target.value).catch((error) => setStatus(error.message, true)));
$("#case-select").addEventListener("change", (event) => { reviewState.selectedCase = event.target.value; renderCase(); });
$("#show-seam").addEventListener("click", showSeam);
initialize();
