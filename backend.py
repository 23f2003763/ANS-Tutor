import io
import os
import uuid
import json
import base64
import secrets
import tempfile
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import ffmpeg
from pydub import AudioSegment
from gtts import gTTS

# STT: faster-whisper (CPU-friendly)
from faster_whisper import WhisperModel

# LLM
import torch
from transformers import pipeline

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# ---------- App & CORS ----------
app = FastAPI(title="ANS Tutor API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: lock down to your domains after you go live
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Simple SQLite (prototype) ----------
DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")
def db_init():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            pw_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chats(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );""")
        con.commit()
db_init()

TOKENS: Dict[str, int] = {}  # token -> user_id

def pw_hash(pw: str) -> str:
    salt = secrets.token_bytes(16)
    import hashlib
    dk = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return base64.b64encode(salt).decode() + ":" + base64.b64encode(dk).decode()

def pw_verify(pw: str, h: str) -> bool:
    try:
        salt_b64, dk_b64 = h.split(":")
        salt = base64.b64decode(salt_b64)
        import hashlib
        cand = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
        return base64.b64encode(cand).decode() == dk_b64
    except Exception:
        return False

def get_user_id_from_token(token: Optional[str]) -> Optional[int]:
    return TOKENS.get(token) if token else None

def auth_dependency(request: Request) -> Optional[int]:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return get_user_id_from_token(token)
    return None

# ---------- Models ----------
class AuthIn(BaseModel):
    email: str
    password: str

class TutorIn(BaseModel):
    session_id: str
    message: str
    original: Optional[str] = ""
    subject: Optional[str] = "math"

# ---------- Session store ----------
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---------- Load LLM ----------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")  # small + stable on CPU
print(f"[LLM] loading {MODEL_ID} on CPU")
GEN = pipeline(
    "text-generation",
    model=MODEL_ID,
    device=-1,
    torch_dtype=torch.float32,
)

# ---------- Load STT ----------
print("[STT] loading faster-whisper (small)")
FW_MODEL = WhisperModel("small", device="cpu", compute_type="int8")  # fast CPU

# ---------- Helpers ----------
def english_system_prompt(subject: str) -> str:
    return (
        f"You are ANS Tutor, a helpful {subject} teacher for JEE Mains. "
        "Explain briefly, write formulas with $...$, and end with a short follow-up question."
    )

def make_write_ops_from_text(text: str) -> List[Dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [{"type": "text", "style": "normal", "content": "(no content)"}]
    ops: List[Dict[str, Any]] = []
    if len(lines[0]) <= 40:
        ops.append({"type": "text", "style": "heading", "content": lines[0]})
        rest = lines[1:]
    else:
        rest = lines
    for ln in rest:
        if "$" in ln:
            ops.append({"type": "formula", "content": ln.replace("$", "")})
        else:
            ops.append({"type": "text", "style": "normal", "content": ln})
    return ops[:12]

def llm_reply(subject: str, user: str) -> str:
    prompt = english_system_prompt(subject) + "\n\nStudent: " + user + "\nTeacher:"
    out = GEN(
        prompt,
        max_new_tokens=220,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        pad_token_id=GEN.tokenizer.eos_token_id,
        eos_token_id=GEN.tokenizer.eos_token_id,
    )
    text = out[0]["generated_text"]
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip().split("\n\n")[0].strip()

def wav_bytes_from_text(text: str) -> bytes:
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3f:
        tts.save(mp3f.name)
        mp3_path = mp3f.name
    try:
        seg = AudioSegment.from_file(mp3_path, format="mp3")
        with io.BytesIO() as buf:
            seg.set_frame_rate(22050).set_channels(1).set_sample_width(2).export(buf, format="wav")
            return buf.getvalue()
    finally:
        try: os.remove(mp3_path)
        except: pass

def webm_bytes_to_wav_bytes(webm_bytes: bytes) -> bytes:
    inbuf = io.BytesIO(webm_bytes); inbuf.seek(0)
    out, _ = (
        ffmpeg
        .input('pipe:', format='webm')
        .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .run(input=inbuf.read(), capture_stdout=True, capture_stderr=True)
    )
    return out

def whisper_transcribe_from_wav_bytes(wav_bytes: bytes) -> Tuple[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        wf.write(wav_bytes)
        wav_path = wf.name
    try:
        segments, info = FW_MODEL.transcribe(wav_path, language=None)
        text = " ".join([s.text for s in segments]).strip()
        lang = info.language or "en"
        return text, lang
    finally:
        try: os.remove(wav_path)
        except: pass

def store_chat(user_id: Optional[int], session_id: str, role: str, content: str):
    if user_id is None: return
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO chats(user_id,session_id,role,content,created_at) VALUES(?,?,?,?,?)",
            (user_id, session_id, role, content, datetime.utcnow().isoformat())
        )
        con.commit()

# ---------- Auth ----------
class SimpleToken(BaseModel):
    token: str

@app.post("/signup")
def signup(data: AuthIn):
    email = data.email.strip().lower()
    if not email or not data.password:
        raise HTTPException(400, "Missing email or password")
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("INSERT INTO users(email, pw_hash, created_at) VALUES(?,?,?)",
                        (email, pw_hash(data.password), datetime.utcnow().isoformat()))
            con.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Email already in use")
    return {"ok": True}

@app.post("/login", response_model=SimpleToken)
def login(data: AuthIn):
    email = data.email.strip().lower()
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT id, pw_hash FROM users WHERE email=?", (email,))
        row = cur.fetchone()
    if not row or not pw_verify(data.password, row[1]):
        raise HTTPException(401, "Invalid credentials")
    token = secrets.token_urlsafe(24)
    TOKENS[token] = row[0]
    return SimpleToken(token=token)

# ---------- Tutor ----------
@app.get("/start_session")
def start_session(subject: str = "math", user_id: Optional[int] = Depends(auth_dependency)):
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"subject": subject, "ops": []}
    say = "Hello! What would you like to learn?"
    ops = [
        {"type":"text","style":"heading","content":"Welcome to ANS Tutor"},
        {"type":"text","style":"normal","content":"Ask any JEE Mains topic or problem."}
    ]
    store_chat(user_id, sid, "tutor", say)
    return {"session_id": sid, "say": say, "write_ops": ops}

@app.post("/stt")
async def stt(session_id: str, audio: UploadFile = File(...), user_id: Optional[int] = Depends(auth_dependency)):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Unknown session")
    raw = await audio.read()
    try:
        wav_bytes = webm_bytes_to_wav_bytes(raw)
        text, lang = whisper_transcribe_from_wav_bytes(wav_bytes)
    except Exception as e:
        raise HTTPException(500, f"STT decode/transcribe error: {e}")
    if text:
        store_chat(user_id, session_id, "student", text)
    return {"text": text, "lang": lang}

class TutorOut(BaseModel):
    say: str
    write_ops: List[Dict[str, Any]]

@app.post("/tutor_turn", response_model=TutorOut)
def tutor_turn(data: TutorIn, user_id: Optional[int] = Depends(auth_dependency)):
    if data.session_id not in SESSIONS:
        raise HTTPException(404, "Unknown session")
    subject = SESSIONS[data.session_id].get("subject", data.subject or "math")
    student_msg = (data.message or "").strip()
    if not student_msg:
        return TutorOut(say="I didn’t catch that—please try again.",
                        write_ops=[{"type":"text","style":"normal","content":"Please try again."}])
    say = llm_reply(subject, student_msg)
    if not say or len(say) < 2:
        say = "Let’s try that again. Could you rephrase your question?"
    write_ops = make_write_ops_from_text(say)
    store_chat(user_id, data.session_id, "tutor", say)
    return TutorOut(say=say, write_ops=write_ops)

@app.get("/tts")
def tts(text: str):
    wav = wav_bytes_from_text(text)
    return StreamingResponse(io.BytesIO(wav), media_type="audio/wav")

@app.get("/export_pdf")
def export_pdf(session_id: str, user_id: Optional[int] = Depends(auth_dependency)):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 2*cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, "ANS Tutor — Session Notes")
    y -= 1.2*cm
    c.setFont("Helvetica", 11)

    if user_id is not None:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("SELECT role, content, created_at FROM chats WHERE user_id=? AND session_id=? ORDER BY id ASC",
                        (user_id, session_id))
            rows = cur.fetchall()
        for role, content, created in rows:
            wrapped = []
            for ln in (content or "").splitlines():
                while len(ln) > 96:
                    wrapped.append(ln[:96]); ln = ln[96:]
                wrapped.append(ln)
            c.drawString(2*cm, y, f"{created}  {role.upper()}:")
            y -= 0.7*cm
            for ln in wrapped:
                if y < 2*cm:
                    c.showPage(); y = H - 2*cm; c.setFont("Helvetica", 11)
                c.drawString(2.4*cm, y, ln)
                y -= 0.6*cm
    else:
        c.drawString(2*cm, y, "Login to export complete chat history.")
        y -= 0.6*cm

    c.showPage()
    c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf")
