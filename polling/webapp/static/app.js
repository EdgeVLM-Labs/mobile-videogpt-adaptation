// Mobile-VideoGPT Coach — patient-first frontend (vanilla JS, no build step)
const $ = (id) => document.getElementById(id);
let es = null;           // EventSource for feedback
let lastSpoken = "";

// ---- element refs ----
const preview = $("preview"), placeholder = $("videoPlaceholder"), statusPill = $("statusPill");
const startBtn = $("startBtn"), stopBtn = $("stopBtn"), voiceToggle = $("voiceToggle");
const fbCard = $("feedbackCard"), fbExercise = $("fbExercise"), fbText = $("fbText"), fbMeta = $("fbMeta");
const historyList = $("historyList");

// ---- advanced drawer ----
const drawer = $("drawer"), overlay = $("overlay");
function openDrawer(){ drawer.classList.remove("hidden"); overlay.classList.remove("hidden"); }
function closeDrawer(){ drawer.classList.add("hidden"); overlay.classList.add("hidden"); }
$("gearBtn").onclick = openDrawer;
$("closeDrawer").onclick = closeDrawer;
overlay.onclick = closeDrawer;

$("sourceMode").onchange = (e) => {
  const file = e.target.value === "file";
  $("fileField").classList.toggle("hidden", !file);
  $("cameraField").classList.toggle("hidden", file);
};

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

  try {
    const vids = (await (await fetch("/api/sample_videos")).json()).videos || [];
    $("videoSelect").innerHTML = vids.map(v => `<option value="${v}">${v}</option>`).join("");
  } catch(e){ console.warn(e); }
}
init();

// ---- status pill ----
function setStatus(state, message){
  statusPill.className = "status-pill " + (state || "idle");
  statusPill.textContent = message || state || "Idle";
}

// ---- feedback rendering ----
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
  fbText.innerHTML = 'Analyzing your form<span class="dots"></span>';
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

// ---- start/stop ----
function payload(){
  const isFile = $("sourceMode").value === "file";
  return {
    is_file: isFile,
    source: isFile ? $("videoSelect").value : $("cameraSelect").value,
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

async function start(){
  startBtn.disabled = true; stopBtn.disabled = false;
  placeholder.classList.add("hidden");
  setStatus("connecting", "Starting…");
  lastSpoken = "";
  // live MJPEG preview (cache-bust so the stream (re)connects).
  // Auto-retry if the long-lived stream drops, so it never sticks on a broken image.
  preview.onerror = () => { setTimeout(() => { preview.src = "/api/preview.mjpg?t=" + Date.now(); }, 1000); };
  preview.src = "/api/preview.mjpg?t=" + Date.now();

  const res = await (await fetch("/api/start", {
    method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload())
  })).json();
  if(!res.ok){ setStatus("error", res.message || "Could not start"); startBtn.disabled=false; stopBtn.disabled=true; return; }

  if(es) es.close();
  es = new EventSource("/api/stream");
  es.onmessage = (m) => {
    let ev; try { ev = JSON.parse(m.data); } catch { return; }
    if(ev.type === "status"){
      setStatus(ev.state, ev.message);
      if(ev.state === "complete"){ onStopped(); }
    } else if(ev.type === "thinking"){
      setStatus("running", "Analyzing…");
      showThinking(ev.poll);
    } else if(ev.type === "partial"){
      setStatus("running", "Coaching…");
      showPartial(ev);
    } else if(ev.type === "feedback"){
      setStatus("running", "Coaching…");
      renderFeedback(ev);
    }
  };
  es.onerror = () => { /* browser auto-reconnects via retry */ };
}

async function stop(){
  await fetch("/api/stop", {method:"POST"});
  onStopped();
}

function onStopped(){
  startBtn.disabled = false; stopBtn.disabled = true;
  if(es){ es.close(); es = null; }
  window.speechSynthesis && window.speechSynthesis.cancel();
}

startBtn.onclick = start;
stopBtn.onclick = stop;
