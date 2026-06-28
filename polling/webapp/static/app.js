// Mobile-VideoGPT Coach — patient-first frontend (vanilla JS, no build step)
// Two modes share one feedback panel + one SSE stream:
//   * Live Coaching   — webcam, MJPEG preview
//   * Analyze a Video — drag/drop or browse a file, then run inference over it
// The stage has three views, driven by #stage[data-view]:
//   "live"  |  "upload-empty" (dropzone)  |  "upload-ready" (player + results)
const $ = (id) => document.getElementById(id);
let es = null;            // EventSource for feedback
let lastSpoken = "";
let currentMode = "live"; // "live" | "upload"
let running = false;
let uploadedName = null;  // server-side filename of the uploaded video

// ---- element refs ----
const stage = $("stage");
const preview = $("preview"), placeholder = $("videoPlaceholder"), statusPill = $("statusPill");
const startBtn = $("startBtn"), stopBtn = $("stopBtn"), voiceToggle = $("voiceToggle");
const fbCard = $("feedbackCard"), fbExercise = $("fbExercise"), fbText = $("fbText"), fbMeta = $("fbMeta");
const historyList = $("historyList");
// tabs
const tabLive = $("tabLive"), tabUpload = $("tabUpload");
// upload
const uploadDropzone = $("uploadDropzone"), fileInput = $("fileInput"),
      changeVideoBtn = $("changeVideoBtn"), analyzeBtn = $("analyzeBtn"),
      analyzeStopBtn = $("analyzeStopBtn"), uploadVideo = $("uploadVideo"),
      uploadStatusPill = $("uploadStatusPill"), uploadHint = $("uploadHint");

// ---- advanced drawer ----
const drawer = $("drawer"), overlay = $("overlay");
function openDrawer(){ drawer.classList.remove("hidden"); overlay.classList.remove("hidden"); }
function closeDrawer(){ drawer.classList.add("hidden"); overlay.classList.add("hidden"); }
$("gearBtn").onclick = openDrawer;
$("closeDrawer").onclick = closeDrawer;
overlay.onclick = closeDrawer;

// ---- light/dark theme toggle ----
// Initial theme is set pre-paint by the inline script in <head>.
const themeBtn = $("themeBtn");
function syncThemeIcon(){
  const dark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
  themeBtn.textContent = dark ? "☀" : "☾";  // icon = the mode you'll switch TO
  themeBtn.title = dark ? "Switch to light theme" : "Switch to dark theme";
}
themeBtn.onclick = () => {
  const next = (document.documentElement.getAttribute("data-theme") === "light") ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("mvgpt-theme", next); } catch(e){}
  syncThemeIcon();
};
syncThemeIcon();

// ---- load defaults + options ----
async function init(){
  try {
    const cfg = await (await fetch("/api/config")).json();
    $("pollingInterval").value = cfg.polling_interval;
    $("fps").value = cfg.fps;
    $("numFrames").value = cfg.num_frames;
    $("maxTokens").value = cfg.max_new_tokens;
    $("baseModel").value = cfg.base_model_path;
    $("loraWeights").value = cfg.lora_weights_path;
    $("prompt").value = cfg.prompt;
  } catch(e){ console.warn(e); }
  try {
    const cams = (await (await fetch("/api/cameras")).json()).cameras || [];
    $("cameraSelect").innerHTML = cams.map(c => `<option value="${c.index}">${c.name}</option>`).join("");
  } catch(e){ console.warn(e); }
}
init();

// ---- status pill (writes to the active mode's pill) ----
function activePill(){ return currentMode === "upload" ? uploadStatusPill : statusPill; }
function setStatus(state, message){
  const p = activePill();
  p.className = "status-pill " + (state || "idle");
  p.textContent = message || state || "Idle";
}

// ---- feedback rendering (shared by both modes) ----
function speak(text){
  if(!voiceToggle.checked || !text || text === lastSpoken) return;
  if(!("speechSynthesis" in window)) return;
  lastSpoken = text;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.0; u.pitch = 1.0;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}

function parsePartial(raw){
  raw = (raw || "").trim();
  const i = raw.indexOf(" - ");
  if(i >= 0) return {ex: raw.slice(0, i).trim(), fb: raw.slice(i + 3).trim()};
  return {ex: raw, fb: ""};
}

function showThinking(poll){
  fbCard.className = "feedback-card state-thinking";
  fbExercise.textContent = "";
  fbText.innerHTML = 'Analyzing the movement<span class="dots"></span>';
  fbMeta.textContent = "Poll #" + poll;
}

function showPartial(ev){
  fbCard.className = "feedback-card state-typing";
  const raw = ev.raw || "";
  if(/^no recognized/i.test(raw)){
    fbExercise.textContent = "…";
    fbText.innerHTML = raw + '<span class="caret">▌</span>';
  } else {
    const {ex, fb} = parsePartial(raw);
    fbExercise.textContent = ex || "…";
    fbText.innerHTML = (fb || "") + '<span class="caret">▌</span>';
  }
  fbMeta.textContent = "Poll #" + ev.poll;
}

function renderFeedback(ev){
  const state = ev.state || "none";
  fbCard.className = "feedback-card state-" + state;
  if(state === "no_exercise"){
    fbExercise.textContent = "No exercise detected";
    fbText.textContent = "Position yourself and start a supported exercise.";
  } else {
    fbExercise.textContent = ev.exercise || "—";
    fbText.textContent = ev.display || ev.feedback || "";
  }
  let meta = `Poll #${ev.poll}`;
  if(ev.latency_ms) meta += ` · ${(ev.latency_ms/1000).toFixed(1)}s total`;
  if(ev.ttft_ms)    meta += ` · TTFT ${(ev.ttft_ms/1000).toFixed(1)}s`;
  if(ev.frames_ms)  meta += ` · frames ${Math.round(ev.frames_ms)}ms`;
  fbMeta.textContent = meta;

  const li = document.createElement("li");
  const ex = (state === "no_exercise") ? "No exercise" : (ev.exercise || "");
  li.innerHTML = `<span class="h-ex">${ex}</span> <span class="h-fb">${ev.display || ev.feedback || ""}</span>`;
  historyList.prepend(li);
  while(historyList.children.length > 8) historyList.removeChild(historyList.lastChild);

  const toSay = (state === "no_exercise") ? "No exercise detected"
              : [ev.exercise, (ev.display || ev.feedback)].filter(Boolean).join(", ");
  speak(toSay);
}

function resetFeedback(){
  fbCard.className = "feedback-card state-idle";
  fbExercise.textContent = "—";
  fbText.textContent = "Waiting to start…";
  fbMeta.textContent = "";
  historyList.innerHTML = "";
  lastSpoken = "";
}

// ---- settings (from the Advanced drawer) — shared by both modes ----
function settings(){
  return {
    polling_interval: parseFloat($("pollingInterval").value),
    fps: parseInt($("fps").value),
    max_new_tokens: parseInt($("maxTokens").value),
    prompt: $("prompt").value,
    base_model_path: $("baseModel").value,
    lora_weights_path: $("loraWeights").value,
    use_naturalizer: $("naturalizer").checked,
    warmup_runs: parseInt($("warmup").value),
  };
}

// ---- SSE wiring (shared) ----
function connectSSE(){
  if(es) es.close();
  es = new EventSource("/api/stream");
  es.onmessage = (m) => {
    let ev; try { ev = JSON.parse(m.data); } catch { return; }
    if(ev.type === "status"){
      setStatus(ev.state, ev.message);
      if(ev.state === "complete"){ onStopped(); }
    } else if(ev.type === "thinking"){ setStatus("running", "Analyzing…"); showThinking(ev.poll); }
    else if(ev.type === "partial"){ setStatus("running", "Coaching…"); showPartial(ev); }
    else if(ev.type === "feedback"){ setStatus("running", "Coaching…"); renderFeedback(ev); }
  };
  es.onerror = () => { /* browser auto-reconnects via retry */ };
}

async function postStart(payload){
  return (await (await fetch("/api/start", {
    method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)
  })).json());
}

// ---- LIVE mode ----
async function startLive(){
  running = true;
  startBtn.disabled = true; stopBtn.disabled = false;
  placeholder.classList.add("hidden");
  setStatus("connecting", "Starting…");
  lastSpoken = "";
  preview.onerror = () => { setTimeout(() => { preview.src = "/api/preview.mjpg?t=" + Date.now(); }, 1000); };
  preview.src = "/api/preview.mjpg?t=" + Date.now();

  const res = await postStart({ is_file:false, source: ($("cameraSelect").value || "0"), ...settings() });
  if(!res.ok){ setStatus("error", res.message || "Could not start"); running=false; startBtn.disabled=false; stopBtn.disabled=true; return; }
  connectSSE();
}
startBtn.onclick = startLive;

// ---- UPLOAD mode ----
function setView(v){ stage.setAttribute("data-view", v); }

// dropzone: click to browse + drag & drop
uploadDropzone.onclick = () => fileInput.click();
fileInput.onchange = () => { if(fileInput.files[0]) handleFile(fileInput.files[0]); };
["dragenter","dragover"].forEach(ev =>
  uploadDropzone.addEventListener(ev, e => { e.preventDefault(); uploadDropzone.classList.add("dragover"); }));
["dragleave","drop"].forEach(ev =>
  uploadDropzone.addEventListener(ev, e => { e.preventDefault(); uploadDropzone.classList.remove("dragover"); }));
uploadDropzone.addEventListener("drop", e => {
  const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if(f) handleFile(f);
});

async function handleFile(f){
  if(f.type && !f.type.startsWith("video/")){ alert("Please choose a video file."); return; }
  uploadedName = null;
  setView("upload-ready");
  uploadVideo.src = URL.createObjectURL(f);
  resetFeedback();
  analyzeBtn.disabled = true; changeVideoBtn.disabled = false;
  uploadStatusPill.className = "status-pill loading"; uploadStatusPill.textContent = "Uploading…";
  uploadHint.textContent = `${f.name} · ${(f.size/1e6).toFixed(1)} MB`;
  try {
    // raw-body upload (no multipart): the file bytes ARE the request body
    const res = await (await fetch("/api/upload?filename=" + encodeURIComponent(f.name), {
      method:"POST", headers:{"Content-Type":"application/octet-stream"}, body: f
    })).json();
    if(!res.ok) throw new Error(res.message || "upload failed");
    uploadedName = res.filename;
    uploadStatusPill.className = "status-pill running"; uploadStatusPill.textContent = "Ready";
    analyzeBtn.disabled = false;
  } catch(e){
    uploadStatusPill.className = "status-pill error"; uploadStatusPill.textContent = "Upload failed";
    uploadHint.textContent = String(e);
  }
}

async function startUpload(){
  if(!uploadedName) return;
  running = true;
  analyzeBtn.disabled = true; analyzeStopBtn.disabled = false; changeVideoBtn.disabled = true;
  setStatus("connecting", "Analyzing…");
  lastSpoken = "";
  try { uploadVideo.currentTime = 0; uploadVideo.play(); } catch(e){}

  const res = await postStart({ is_file:true, source: uploadedName, ...settings() });
  if(!res.ok){
    setStatus("error", res.message || "Could not start");
    running=false; analyzeBtn.disabled=false; analyzeStopBtn.disabled=true; changeVideoBtn.disabled=false; return;
  }
  connectSSE();
}
analyzeBtn.onclick = startUpload;

changeVideoBtn.onclick = () => {
  if(running) stopSession();
  uploadedName = null;
  try { uploadVideo.pause(); } catch(e){}
  uploadVideo.removeAttribute("src"); uploadVideo.load();
  fileInput.value = "";
  resetFeedback();
  uploadStatusPill.className = "status-pill idle"; uploadStatusPill.textContent = "Ready";
  setView("upload-empty");
};

// ---- stop / completion (shared) ----
async function stopSession(){ try { await fetch("/api/stop", {method:"POST"}); } catch(e){} onStopped(); }
function onStopped(){
  running = false;
  startBtn.disabled = false; stopBtn.disabled = true;
  analyzeStopBtn.disabled = true; changeVideoBtn.disabled = false;
  analyzeBtn.disabled = !uploadedName;
  if(es){ es.close(); es = null; }
  window.speechSynthesis && window.speechSynthesis.cancel();
}
stopBtn.onclick = stopSession;
analyzeStopBtn.onclick = stopSession;

// ---- mode switching ----
function setMode(mode){
  if(mode === currentMode) return;
  if(running) stopSession();
  currentMode = mode;
  const live = mode === "live";
  tabLive.classList.toggle("active", live);
  tabUpload.classList.toggle("active", !live);
  if(live){
    setView("live");
  } else {
    preview.onerror = null; preview.removeAttribute("src");  // free the MJPEG stream
    setView(uploadedName ? "upload-ready" : "upload-empty");
  }
  resetFeedback();
}
tabLive.onclick = () => setMode("live");
tabUpload.onclick = () => setMode("upload");
