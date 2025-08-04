// ===== Config =====
const API = (location.hostname === "127.0.0.1" || location.hostname === "localhost")
  ? "http://127.0.0.1:8000"
  : (location.origin.replace(/\/$/, "")); // when deployed behind same domain/proxy

// Fonts per board language (English only in this MVP)
const FONT_URLS = { en: "./fonts/PatrickHand-Regular.ttf" };

// ===== State =====
let chat, penLayer, penCursor, statusEl, whoEl, startBtn, speakBtn, downloadPdfBtn, textInput, sendBtn, toggleChatBtn, chatPanel, closeChatBtn, subjectEl;
let loginBtn, authModal, emailEl, passwordEl, signupBtn, signinBtn, closeAuthBtn, authMsgEl;

let font = null;
let sessionId = null;
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let speaking = false;
let busy = false;
let audioRef = null;
let jwtToken = null;  // simple token stored after login
let currentSubject = "math";

// ===== Utilities =====
function bindDom() {
  chat = document.getElementById("chat");
  penLayer = document.getElementById("pen");
  penCursor = document.getElementById("penCursor");
  statusEl = document.getElementById("status");
  whoEl = document.getElementById("who");
  startBtn = document.getElementById("startBtn");
  speakBtn = document.getElementById("speakBtn");
  downloadPdfBtn = document.getElementById("downloadPdf");
  textInput = document.getElementById("textInput");
  sendBtn = document.getElementById("sendBtn");
  toggleChatBtn = document.getElementById("toggleChat");
  chatPanel = document.getElementById("chatPanel");
  closeChatBtn = document.getElementById("closeChat");
  subjectEl = document.getElementById("subject");

  loginBtn = document.getElementById("loginBtn");
  authModal = document.getElementById("authModal");
  emailEl = document.getElementById("email");
  passwordEl = document.getElementById("password");
  signupBtn = document.getElementById("signup");
  signinBtn = document.getElementById("signin");
  closeAuthBtn = document.getElementById("closeAuth");
  authMsgEl = document.getElementById("authMsg");
}

function attachListeners() {
  if (!startBtn) return;
  toggleChatBtn.addEventListener("click", () => (chatPanel.style.display = "flex"));
  closeChatBtn.addEventListener("click", () => (chatPanel.style.display = "none"));
  startBtn.addEventListener("click", startSession);
  speakBtn.addEventListener("click", toggleSpeak);
  sendBtn.addEventListener("click", sendText);

  downloadPdfBtn.addEventListener("click", downloadPdf);

  subjectEl.addEventListener("change", () => {
    currentSubject = subjectEl.value || "math";
  });

  loginBtn.addEventListener("click", () => { authModal.style.display = "flex"; authMsgEl.textContent = ""; });
  closeAuthBtn.addEventListener("click", () => { authModal.style.display = "none"; });
  signupBtn.addEventListener("click", authSignup);
  signinBtn.addEventListener("click", authSignin);
}

function addMsg(text, role = "tutor") {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}
function setStatus(s) { statusEl.textContent = s; }
function setWho(s) { whoEl.textContent = s; }
function enableControls(on = true) {
  speakBtn.disabled = !on;
  sendBtn.disabled = !on;
  textInput.disabled = !on;
  downloadPdfBtn.disabled = !sessionId;
}

// Lazy-load opentype if not present
function ensureOpenType() {
  if (window.opentype) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/opentype.js@1.3.4/dist/opentype.min.js";
    s.defer = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load opentype.js"));
    document.head.appendChild(s);
  });
}

async function loadFont() {
  await ensureOpenType();
  const url = FONT_URLS.en;
  return new Promise((res, rej) => {
    opentype.load(url, (err, f) => (err ? rej(err) : res(f)));
  });
}
async function ensureFont() {
  if (!font) {
    try { font = await loadFont(); }
    catch(e){ console.warn("Font load failed", e); }
  }
}

function wrapTextByWidth(text, fontSize, maxWidth) {
  const safe = typeof text === "string" ? text : "";
  const words = safe.split(/\s+/);
  const lines = [];
  let line = "";
  const measure = (str) => font.getAdvanceWidth(str, fontSize, { kerning: true });
  for (const w of words) {
    const t = line ? line + " " + w : w;
    if (measure(t) <= maxWidth) line = t;
    else {
      if (line) lines.push(line);
      if (measure(w) > maxWidth) {
        let cur = "";
        for (const ch of w) {
          if (measure(cur + ch) <= maxWidth) cur += ch;
          else { lines.push(cur); cur = ch; }
        }
        line = cur;
      } else line = w;
    }
  }
  if (line) lines.push(line);
  return lines;
}
function mkPath(text, x, y, size = 40) {
  const p = font.getPath(text || "", x, y, size, { kerning: true });
  return p.toPathData(2);
}

// Pen cursor helpers
function movePenCursorFor(pathEl, r) {
  try {
    const pathBox = pathEl.getBoundingClientRect();
    const layerBox = penLayer.getBoundingClientRect();
    const x = pathBox.left - layerBox.left + pathBox.width * r - 12; // larger nib offset
    const y = pathBox.top - layerBox.top + pathBox.height * 0.5 - 12;
    penCursor.style.transform = `translate(${x}px, ${y}px)`;
  } catch {}
}
function hidePenCursor() { penCursor.style.transform = "translate(-9999px, -9999px)"; }

async function drawFilledPath(d, cls = "pen-path", height = 90, duration = 700) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "pen-svg");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", String(height));
  const path = document.createElementNS(NS, "path");
  path.setAttribute("d", d);
  path.setAttribute("class", cls);
  svg.appendChild(path);
  penLayer.appendChild(svg);
  svg.scrollIntoView({ behavior: "smooth", block: "end" });

  path.style.clipPath = `polygon(0 0, 0 0, 0 100%, 0 100%)`;
  const start = performance.now();
  await new Promise((res) => {
    function step(t) {
      const r = Math.min(1, (t - start) / duration);
      const x = (r * 100).toFixed(2) + "%";
      path.style.clipPath = `polygon(0 0, ${x} 0, ${x} 100%, 0 100%)`;
      movePenCursorFor(path, r);
      if (r < 1) requestAnimationFrame(step);
      else res();
    }
    requestAnimationFrame(step);
  });
  hidePenCursor();
}

async function penWriteText(text, style = "normal") {
  await ensureFont();
  const content = typeof text === "string" && text.trim().length ? text : " ";
  const size = style === "heading" ? 56 : 42;
  const lineH = size * 1.8;
  const maxW = Math.max(200, penLayer.clientWidth - 48);
  const lines = wrapTextByWidth(content, size, maxW - 48);
  for (const ln of lines) {
    const d = mkPath(ln, 24, lineH - 18, size);
    await drawFilledPath(d, style === "heading" ? "pen-path heading" : "pen-path", lineH, Math.max(500, 60 * ln.length));
  }
}
async function penWriteFormula(tex) {
  const box = document.createElement("div");
  box.className = "formula-box";
  penLayer.appendChild(box);
  box.scrollIntoView({ behavior: "smooth", block: "end" });
  await MathJax.typesetPromise([box]);
  box.innerHTML = "";
  const host = document.createElement("span");
  box.appendChild(host);
  MathJax.tex2svgPromise("\\displaystyle " + (tex || "")).then((svgNode) => {
    const svg = svgNode.querySelector("svg");
    svg.style.width = "100%";
    svg.style.height = "auto";
    svg.querySelectorAll("path").forEach((p) => {
      p.style.fill = "none";
      p.style.stroke = "var(--ink-formula)";
      p.style.strokeWidth = "2.6px";
      const len = p.getTotalLength();
      p.style.strokeDasharray = len;
      p.style.strokeDashoffset = len;
    });
    host.appendChild(svg);
    (async () => {
      for (const p of svg.querySelectorAll("path")) {
        const len = p.getTotalLength();
        const dur = Math.max(120, Math.min(900, len * 1.0));
        const t0 = performance.now();
        await new Promise((done) => {
          function step(t) {
            const r = Math.min(1, (t - t0) / dur);
            p.style.strokeDashoffset = String(len * (1 - r));
            if (r < 1) requestAnimationFrame(step);
            else done();
          }
          requestAnimationFrame(step);
        });
      }
    })();
  });
}
function penWriteCode(code) {
  const pre = document.createElement("pre");
  pre.className = "code";
  pre.textContent = code || "";
  penLayer.appendChild(pre);
  pre.scrollIntoView({ behavior: "smooth", block: "end" });
}
function penDiagram(obj) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "pen-svg");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", String(obj.h || 220));
  const stroke = getComputedStyle(document.documentElement).getPropertyValue("--ink");
  const sw = 2.4;
  function line(x1, y1, x2, y2) {
    const L = document.createElementNS(NS, "line");
    L.setAttribute("x1", x1); L.setAttribute("y1", y1);
    L.setAttribute("x2", x2); L.setAttribute("y2", y2);
    L.setAttribute("stroke", stroke); L.setAttribute("stroke-width", sw);
    return L;
  }
  function text(x, y, t) {
    const T = document.createElementNS(NS, "text");
    T.setAttribute("x", x); T.setAttribute("y", y);
    T.setAttribute("fill", stroke); T.setAttribute("font-size", "16");
    T.textContent = t || "";
    return T;
  }
  if (obj.type === "triangle") {
    const x = obj.x || 60, y = obj.y || 60, a = obj.a || 150, b = obj.b || 100;
    svg.appendChild(line(x, y + b, x + a, y + b));
    svg.appendChild(line(x, y + b, x, y));
    svg.appendChild(line(x, y, x + a, y + b));
    if (obj.label_a) svg.appendChild(text(x + a / 2, y + b + 18, obj.label_a));
    if (obj.label_b) svg.appendChild(text(x - 18, y + b / 2, obj.label_b));
    if (obj.label_c) svg.appendChild(text(x + a / 2 - 10, y + b / 2 - 6, obj.label_c));
  }
  penLayer.appendChild(svg);
  svg.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function applyWriteOps(ops) {
  for (const op of ops || []) {
    if (!op || !op.type) continue;
    if (op.type === "text") await penWriteText(op.content || "", op.style === "heading" ? "heading" : "normal");
    else if (op.type === "formula") await penWriteFormula(op.content || "");
    else if (op.type === "code") penWriteCode(op.content || "");
    else penDiagram(op);
  }
}

// ===== Audio (TTS) =====
async function speakFetchThenPlay(text) {
  if (!text || !text.trim()) {
    setStatus("No speech");
    return { audio: null };
  }
  const headers = jwtToken ? { Authorization: `Bearer ${jwtToken}` } : {};
  const res = await fetch(`${API}/tts?text=${encodeURIComponent(text)}`, { mode: "cors", headers });
  if (!res.ok) {
    setStatus(`TTS ${res.status}`);
    return { audio: null };
  }
  const ct = (res.headers.get("content-type") || "");
  if (!ct.includes("audio/wav")) {
    setStatus("TTS non-audio");
    return { audio: null };
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  await new Promise((r) => { audio.addEventListener("loadedmetadata", r, { once: true }); audio.load(); });
  audioRef = audio;
  audio.play().catch(() => {});
  return { audio };
}

// ===== STT / Tutor =====
async function uploadAudio(blob) {
  const form = new FormData();
  form.append("audio", blob, "speech.webm");
  const headers = jwtToken ? { Authorization: `Bearer ${jwtToken}` } : {};
  const res = await fetch(`${API}/stt?session_id=${encodeURIComponent(sessionId)}`, {
    method: "POST",
    body: form,
    mode: "cors",
    headers
  });
  if (!res.ok) throw new Error("STT " + res.status);
  return res.json(); // {text, lang}
}
async function tutorTurn(message, original) {
  const headers = { "Content-Type": "application/json" };
  if (jwtToken) headers["Authorization"] = `Bearer ${jwtToken}`;
  const res = await fetch(`${API}/tutor_turn`, {
    method: "POST",
    headers,
    body: JSON.stringify({ session_id: sessionId, message, original, subject: currentSubject }),
    mode: "cors",
  });
  if (!res.ok) throw new Error("tutor_turn " + res.status);
  return res.json();
}

// ===== Orchestration =====
async function runTeacherTurn(say, write_ops) {
  busy = true;
  enableControls(false);
  setWho("Teacher speaking…");
  const { audio } = await speakFetchThenPlay(say);
  addMsg(say, "tutor");
  await applyWriteOps(write_ops);

  // Safety: re-enable even if 'ended' never fires (autoplay blocked etc.)
  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    busy = false;
    enableControls(true);
    setWho("Idle");
    setStatus("Your turn");
  };

  if (audio) {
    audio.addEventListener("ended", finish, { once: true });
    // fallback timeout
    setTimeout(finish, Math.min(30000, Math.max(5000, Math.floor(audio.duration * 1000) || 8000)));
  } else {
    // no audio received
    setTimeout(finish, 1500);
  }
}

// ===== Push-to-talk =====
let mediaType = null;
async function toggleSpeak() {
  if (!sessionId) { setStatus("Click Start & Greet first"); return; }
  if (busy) { setStatus("Please wait for the teacher"); return; }

  if (speaking) {
    speaking = false;
    speakBtn.textContent = "Speak";
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop(); setStatus("Processing…"); setWho("Teacher thinking…");
    }
    return;
  }

  speaking = true; speakBtn.textContent = "Stop";
  setStatus("Listening…"); setWho("Student speaking…");

  if (!mediaStream) {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  }
  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
  mediaType = mime;
  mediaRecorder = new MediaRecorder(mediaStream, { mimeType: mime });
  recordedChunks = [];
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = async () => {
    try {
      const blob = new Blob(recordedChunks, { type: mediaType }); recordedChunks = [];
      const { text } = await uploadAudio(blob);
      if (text && text.trim()) {
        addMsg(text.trim(), "student");
        setStatus("Thinking…"); setWho("Teacher thinking…");
        const { say, write_ops } = await tutorTurn(text.trim(), "");
        await runTeacherTurn(say, write_ops);
      } else {
        setStatus("I didn’t catch that—please try again.");
        setWho("Idle");
        enableControls(true);
      }
    } catch (e) {
      console.error(e);
      setStatus("STT/Tutor error");
      setWho("Idle");
      enableControls(true);
    }
  };
  mediaRecorder.start();
}

// ===== Send text =====
async function sendText() {
  if (!sessionId) { setStatus("Click Start & Greet first"); return; }
  if (busy) { setStatus("Please wait for the teacher"); return; }
  const t = textInput.value.trim(); if (!t) return;
  textInput.value = ""; addMsg(t, "student"); setStatus("Thinking…"); setWho("Teacher thinking…");
  try {
    const { say, write_ops } = await tutorTurn(t, t);
    await runTeacherTurn(say, write_ops);
  } catch (e) {
    console.error(e);
    setStatus("Tutor error");
    setWho("Idle");
  }
}

// ===== PDF =====
function downloadPdf() {
  if (!sessionId) return;
  const a = document.createElement("a");
  a.href = `${API}/export_pdf?session_id=${encodeURIComponent(sessionId)}`;
  a.download = `tutor_${sessionId}.pdf`;
  document.body.appendChild(a); a.click(); a.remove();
}

// ===== Start / events =====
async function startSession() {
  setStatus("Starting…"); enableControls(false); setWho("Waiting…");
  try {
    await ensureOpenType();
    await ensureFont();
    const headers = jwtToken ? { Authorization: `Bearer ${jwtToken}` } : {};
    const url = `${API}/start_session?subject=${encodeURIComponent(currentSubject)}`;
    const r = await fetch(url, { mode: "cors", headers });
    if (!r.ok) throw new Error("start_session " + r.status);
    const data = await r.json();
    sessionId = data.session_id;
    await runTeacherTurn(data.say || "Hello! What would you like to learn?", data.write_ops || []);
  } catch (e) {
    console.error(e);
    setStatus("Start failed");
    enableControls(true);
    setWho("Idle");
  }
}

// ===== Auth =====
async function authSignup() {
  authMsgEl.textContent = "Signing up…";
  const r = await fetch(`${API}/signup`, {
    method: "POST",
    headers: { "Content-Type":"application/json" },
    body: JSON.stringify({ email: emailEl.value.trim(), password: passwordEl.value })
  });
  const j = await r.json().catch(()=> ({}));
  if (!r.ok) { authMsgEl.textContent = j.detail || "Signup failed"; return; }
  authMsgEl.textContent = "Account created. You can log in now.";
}
async function authSignin() {
  authMsgEl.textContent = "Logging in…";
  const r = await fetch(`${API}/login`, {
    method: "POST",
    headers: { "Content-Type":"application/json" },
    body: JSON.stringify({ email: emailEl.value.trim(), password: passwordEl.value })
  });
  const j = await r.json().catch(()=> ({}));
  if (!r.ok || !j.token) { authMsgEl.textContent = j.detail || "Login failed"; return; }
  jwtToken = j.token;
  authMsgEl.textContent = "Logged in.";
  authModal.style.display = "none";
  loginBtn.textContent = "Logged in";
  loginBtn.disabled = true;
}

// ===== Init =====
function init() {
  bindDom();
  attachListeners();
  enableControls(false);
  setStatus("Click “Start & Greet”");
  setWho("Idle");
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
