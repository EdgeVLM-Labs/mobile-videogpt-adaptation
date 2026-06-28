// Mobile-VideoGPT Coach — patient-first frontend (vanilla JS, no build step)
// Two modes share one feedback panel + one SSE stream:
//   * Live Coaching  — webcam, MJPEG preview
//   * Analyze a Video — upload a file, run inference over it
const $ = (id) => document.getElementById(id);
let es = null;            // EventSource for feedback
let lastSpoken = "";
let currentMode = "live"; // "live" | "upload"
let running = false;
let uploadedName = null;  // server-side filename of the uploaded video

// ---- element refs ----
const preview = $("preview"), placeholder = $("videoPlaceholder"), statusPill = $("statusPill");
const startBtn = $("startBtn"), stopBtn = $("stopBtn"), voiceToggle = $("voiceToggle");
const fbCard = $("feedbackCard"), fbExercise = $("fbExercise"), fbText = $("fbText"), fbMeta = $("fbMeta");
const historyList = $("historyList");
// tabs + columns
const tabLive = $("tabLive"), tabUpload = $("tabUpload"), liveCol = $("liveCol"), uploadCol = $("uploadCol");
// upload controls
const fileInput = $("fileInput"), chooseBtn = $("chooseBtn"),
      analyzeBtn = $("analyzeBtn"), analyzeStopBtn = $("analyzeStopBtn"),
      uploadVideo = $("uploadVideo"), uploadStatusPill = $("uploadStatusPill"),
      uploadPlaceholder = $("uploadPlaceholder"), uploadHint = $("uploadHint");

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

// shown during prefill (model encoding 16 frames, before the first token)
function showThinking(poll){
  fbCard.className = "feedback-card state-thinking";
  fbExercise.textContent = "";
  fbText.innerHTML = 'Analyzing the movement<span class="dots"></span>';
  fbMeta.textContent = "Poll #" + poll;
}

// shown per token as the response streams in
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

  // history (newest first, cap 8)
  const li = document.createElement("li");
  const ex = (state === "no_exercise") ? "No exercise" : (ev.exercise || "");
  li.innerHTML = `<span class="h-ex">${ex}</span> <span class="h-fb">${ev.display || ev.feedback || ""}</span>`;
  historyList.prepend(li);
  while(historyList.children.length > 8) historyList.removeChild(historyList.lastChild);

  // voice: speak the spoken-form feedback
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
  // live MJPEG preview (cache-bust so it (re)connects; auto-retry if it drops)
  preview.onerror = () => { setTimeout(() => { preview.src = "/api/preview.mjpg?t=" + Date.now(); }, 1000); };
  preview.src = "/api/preview.mjpg?t=" + Date.now();

  const res = await postStart({ is_file:false, source: ($("cameraSelect").value || "0"), ...settings() });
  if(!res.ok){ setStatus("error", res.message || "Could not start"); running=false; startBtn.disabled=false; stopBtn.disabled=true; return; }
  connectSSE();
}

// ---- UPLOAD mode ----
chooseBtn.onclick = () => fileInput.click();

fileInput.onchange = async () => {
  const f = fileInput.files[0];
  if(!f) return;
  // instant local preview (no round-trip) while we upload for inference
  uploadVideo.src = URL.createObjectURL(f);
  uploadPlaceholder.classList.add("hidden");
  analyzeBtn.disabled = true; uploadedName = null;
  uploadStatusPill.className = "status-pill loading"; uploadStatusPill.textContent = "Uploading…";
  uploadHint.textContent = `${f.name} · ${(f.size/1e6).toFixed(1)} MB`;
  try {
    // raw-body upload (no multipart): file bytes are the request body
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
};

async function startUpload(){
  if(!uploadedName) return;
  running = true;
  analyzeBtn.disabled = true; analyzeStopBtn.disabled = false; chooseBtn.disabled = true;
  setStatus("connecting", "Analyzing…");
  lastSpoken = "";
  try { uploadVideo.currentTime = 0; uploadVideo.play(); } catch(e){}

  const res = await postStart({ is_file:true, source: uploadedName, ...settings() });
  if(!res.ok){
    setStatus("error", res.message || "Could not start");
    running=false; analyzeBtn.disabled=false; analyzeStopBtn.disabled=true; chooseBtn.disabled=false; return;
  }
  connectSSE();
}

// ---- stop / completion (shared) ----
async function stopSession(){ try { await fetch("/api/stop", {method:"POST"}); } catch(e){} onStopped(); }
function onStopped(){
  running = false;
  startBtn.disabled = false; stopBtn.disabled = true;
  analyzeStopBtn.disabled = true; chooseBtn.disabled = false;
  analyzeBtn.disabled = !uploadedName;
  if(es){ es.close(); es = null; }
  window.speechSynthesis && window.speechSynthesis.cancel();
}

startBtn.onclick = startLive;
stopBtn.onclick = stopSession;
analyzeBtn.onclick = startUpload;
analyzeStopBtn.onclick = stopSession;

// ---- mode switching ----
function setMode(mode){
  if(mode === currentMode) return;
  if(running) stopSession();
  currentMode = mode;
  const live = mode === "live";
  tabLive.classList.toggle("active", live);
  tabUpload.classList.toggle("active", !live);
  liveCol.classList.toggle("hidden", !live);
  uploadCol.classList.toggle("hidden", live);
  if(!live){ preview.onerror = null; preview.removeAttribute("src"); }  // free the MJPEG stream
  resetFeedback();
  setStatus("idle", live ? "Idle" : (uploadedName ? "Ready" : "No file"));
}
tabLive.onclick = () => setMode("live");
tabUpload.onclick = () => setMode("upload");
