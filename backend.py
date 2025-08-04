import io
import os
import uuid
import json
import time
import base64
import secrets
import tempfile
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import ffmpeg
from pydub import AudioSegment
from gtts import gTTS

# STT
import whisper

# LLM
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# ---------- App & CORS ----------
app = FastAPI(title="ANS Tutor API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # when deploying, lock this down to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Simple “DB” (SQLite) ----------
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
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chats(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
        con.commit()
db_init()

# in-memory token store (prototype only)
TOKENS: Dict[str, int] = {}  # token -> user_id

def pw_hash(pw: str) -> str:
    # simple scrypt (prototype)
    salt = secrets.token_bytes(16)
    key = base64.b64encode(os.urandom(32)).decode()
    dk = base64.b64encode(
        bytes(
            hashlib_scrypt(pw.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
        )
    ).decode()
    return base64.b64encode(salt).decode() + ":" + dk

def hashlib_scrypt(pw_bytes: bytes, salt: bytes, n: int, r: int, p: int, dklen: int):
    import hashlib
    return hashlib.scrypt(pw_bytes, salt=salt, n=n, r=r, p=p, dklen=dklen)

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
    if not token: return None
    return TOKENS.get(token)

def auth_dependency(request: Request) -> Optional[int]:
    # return user_id or None
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
SESSIONS: Dict[str, Dict[str, Any]] = {}  # session_id -> state (write_ops, etc.)

# ---------- Load models ----------
DEVICE = "cpu"  # keep CPU for portability; switch to "cuda" on a GPU host
print("[LLM] loading microsoft/Phi-3-mini-4k-instruct on", DEVICE)

# Use pipeline to avoid low-level generate kwargs issues
GEN = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device=0 if DEVICE == "cuda" else -1,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

# STT (english default)
print("[STT] loading whisper-small")
stt_model = whisper.load_model("small", device=DEVICE)

# ---------- Helpers ----------
def english_system_prompt(subject: str) -> str:
    # Short, safe instruction for the teacher persona.
    return (
        f"You are ANS Tutor, a patient {subject} teacher for JEE Mains preparation. "
        "Explain concepts clearly and briefly, then ask a short follow-up question. "
        "Use LaTeX for formulas (like E=mc^2 -> $E=mc^2$)."
    )

def make_write_ops_from_text(text: str) -> List[Dict[str, Any]]:
    # Very simple heuristic: first line as heading if short, rest normal
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    ops: List[Dict[str, Any]] = []
    if not lines:
        return [{"type":"text","style":"normal","content":"(no content)"}]
    if len(lines[0]) <= 40:
        ops.append({"type":"text","style":"heading","content":lines[0]})
        for ln in lines[1:]:
            # extract $...$ math to a separate op
            if "$" in ln:
                ops.append({"type":"formula","content":ln.replace("$","")})
            else:
                ops.append({"type":"text","style":"normal","content":ln})
    else:
        # just paragraphs
        block = " ".join(lines)
        # split by sentences
        chunks = [c.strip() for c in block.split(". ") if c.strip()]
        for ck in chunks[:6]:
            if "$" in ck:
                ops.append({"type":"formula","content":ck.replace("$","")})
            else:
                ops.append({"type":"text","style":"normal","content":ck})
    return ops[:12]

def llm_reply(subject: str, user: str) -> str:
    prompt = english_system_prompt(subject) + "\n\nStudent: " + user + "\nTeacher:"
    out = GEN(
        prompt,
        max_new_tokens=240,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        eos_token_id=GEN.tokenizer.eos_token_id
    )
    text = out[0]["generated_text"]
    # strip the prepended prompt if pipeline returned full text
    if text.startswith(prompt):
        text = text[len(prompt):]
    # keep it short-ish
    return text.strip().split("\n\n")[0].strip()

def wav_bytes_from_text(text: str) -> bytes:
    # gTTS -> mp3 -> wav (pydub)
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
    """Decode browser webm/opus to mono wav via ffmpeg in-memory."""
    inbuf = io.BytesIO(webm_bytes)
    inbuf.seek(0)
    out, _ = (
        ffmpeg
        .input('pipe:', format='webm')
        .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .run(input=inbuf.read(), capture_stdout=True, capture_stderr=True)
    )
    return out

def whisper_transcribe_from_wav_bytes(wav_bytes: bytes) -> Tuple[str, str]:
    """Return (text, lang)."""
    # Write to temp wav, then let Whisper load it. This avoids np/bytes confusion.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        wf.write(wav_bytes)
        wav_path = wf.name
    try:
        result = stt_model.transcribe(wav_path, language=None, fp16=(DEVICE=="cuda"))
        text = (result.get("text") or "").strip()
        lang = (result.get("language") or "en")
        return text, lang
    finally:
        try: os.remove(wav_path)
        except: pass

def store_chat(user_id: Optional[int], session_id: str, role: str, content: str):
    if user_id is None: return
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("INSERT INTO chats(user_id,session_id,role,content,created_at) VALUES(?,?,?,?,?)",
                    (user_id, session_id, role, content, datetime.utcnow().isoformat()))
        con.commit()

# ---------- Auth endpoints ----------
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

@app.post("/login")
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
    return {"token": token}

# ---------- Tutor endpoints ----------
@app.get("/start_session")
def start_session(subject: str = "math", user_id: Optional[int] = Depends(auth_dependency)):
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"subject": subject, "ops": []}
    # English-only greet:
    say = "Hello! What would you like to learn?"
    ops = [{"type":"text","style":"heading","content":"Welcome to ANS Tutor"},
           {"type":"text","style":"normal","content":"Ask any JEE Mains topic or problem."}]
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
    # store student message (if logged in)
    if text:
        store_chat(user_id, session_id, "student", text)
    return {"text": text, "lang": lang}

@app.post("/tutor_turn")
def tutor_turn(data: TutorIn, user_id: Optional[int] = Depends(auth_dependency)):
    if data.session_id not in SESSIONS:
        raise HTTPException(404, "Unknown session")
    subject = SESSIONS[data.session_id].get("subject", data.subject or "math")
    student_msg = data.message.strip()
    if not student_msg:
        return {"say":"I didn’t catch that—please try again.", "write_ops":[{"type":"text","style":"normal","content":"Please try again."}]}
    # LLM reply (English)
    say = llm_reply(subject, student_msg)
    if not say or len(say) < 2:
        say = "Let’s try that again. Could you rephrase your question?"
    write_ops = make_write_ops_from_text(say)
    # store teacher message
    store_chat(user_id, data.session_id, "tutor", say)
    return {"say": say, "write_ops": write_ops}

@app.get("/tts")
def tts(text: str):
    # English-only TTS
    wav = wav_bytes_from_text(text)
    return StreamingResponse(io.BytesIO(wav), media_type="audio/wav")

@app.get("/export_pdf")
def export_pdf(session_id: str, user_id: Optional[int] = Depends(auth_dependency)):
    # naive: write all chat in a PDF (if user logged in, use stored history; otherwise simple export)
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

# ---------- Uvicorn ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=True)
