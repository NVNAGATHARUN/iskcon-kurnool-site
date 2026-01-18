import json
import os
from typing import List, Tuple
import csv
from datetime import datetime
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
CONTACT_CSV = "contact_submissions.csv"


load_dotenv()

CONTACT_CSV = "contact_submissions.csv"

EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "465"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO  = os.getenv("EMAIL_TO")


def send_contact_email(row: dict) -> None:
    """Send a simple email with contact form contents."""
    if not (EMAIL_HOST and EMAIL_USER and EMAIL_PASS and EMAIL_TO):
        # Misconfigured email settings; just skip mail, keep CSV.
        print("Email not sent: EMAIL_* env vars missing")
        return

    subject = f"New ISKCON Kurnool contact from {row['name']}"
    body = (
        f"Time (UTC): {row['timestamp']}\n"
        f"Name      : {row['name']}\n"
        f"Phone     : {row['phone']}\n"
        f"Email     : {row['email']}\n"
        f"Subject   : {row['subject']}\n\n"
        f"Message:\n{row['message']}\n"
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg.set_content(body)

    # Gmail SSL on 465
    with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)

# ===== Config =====
EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"  # lightweight, good for Q&A
INDEX_PATH = "gita_faiss.index"
META_PATH = "gita_meta.json"
TOP_K = 5
# ==================


# Questions that are clearly outside Bhagavad Gita scope
OUT_OF_GITA_KEYWORDS = [
    "jesus", "christ", "bible",
    "muhammad", "mohammed", "quran", "allah",
    "church", "mosque",
    "islam", "christianity", "judaism",
    "prophet", "messiah",
]


def is_out_of_gita_scope(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in OUT_OF_GITA_KEYWORDS)


def get_client() -> genai.Client:
    # Load .env file and read GOOGLE_API_KEY
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment or .env file")
    # Gemini Developer API client
    client = genai.Client(api_key=api_key)  # [web:183][web:189]
    return client


def load_index_and_meta():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


client = get_client()
index, META = load_index_and_meta()

app = FastAPI(title="My Krishna Chatbot")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    references: List[str]
    reference_links: List[str]  # clickable URLs for each reference


def polite_out_of_scope_reply() -> AskResponse:
    answer = (
        "I speak only through the wisdom of the Bhagavad Gita.\n"
        "The Gita does not speak about this topic or person by name.\n"
        "I guide the soul toward duty, devotion, and inner steadiness.\n"
        "If you wish, ask me a question about life, actions, peace, or devotion as taught in the Gita."
    )
    return AskResponse(answer=answer, references=[], reference_links=[])


def embed_text(text: str) -> np.ndarray:
    # Call embeddings API
    result = client.models.embed_content(  # [web:190][web:195]
        model=EMBEDDING_MODEL,
        contents=[{"parts": [{"text": text}]}],
    )
    vec = np.array(result.embeddings[0].values, dtype="float32")
    return vec.reshape(1, -1)


def search_verses(question: str, k: int = TOP_K):
    q_vec = embed_text(question)
    distances, ids = index.search(q_vec, k)
    ids = ids[0]
    results = []
    for i in ids:
        if i == -1:
            continue
        v = META[int(i)]
        results.append(v)
    return results


def build_system_prompt():
    return (
        "You are 'My Krishna', a calm, compassionate spiritual guide inspired by the teachings of Lord Krishna in the Bhagavad Gita.\n\n"
        "Your primary source of knowledge is the Bhagavad Gita verses provided to you. "
        "Do NOT bring in other religious books or modern philosophies; stay within Gita teachings, "
        "but you may paraphrase them in simple words.\n\n"
        "Your role is to help students, families, and seekers find clarity, peace, and right understanding.\n"
        "Speak gently, with warmth and humility, like a caring guide — not as an authority or judge.\n\n"
        "STYLE AND FORMAT (VERY IMPORTANT):\n"
        "Always answer in 3 short blocks, like this:\n"
        "1) First line: connect the user's feeling with Arjuna or a simple understanding sentence.\n"
        "2) Second block: one key line from a relevant Bhagavad Gita verse in simple English, "
        "with its reference, for example: \"In Bhagavad Gita 2.47, I say: 'Focus on your duty, not on the result.'\".\n"
        "3) Third block: 1–2 short sentences of practical instruction for the user (what they can think or do now).\n\n"
        "Language rules:\n"
        "• Use very simple English so even a 15-year-old can understand.\n"
        "• Keep sentences short and clear.\n"
        "• Avoid heavy Sanskrit or complex philosophical terms.\n"
        "• Do not write long paragraphs or stories; keep the whole answer within 4–6 short lines.\n"
        "• Put each sentence on its own line, like a chat message.\n"
        "• Always mention the Gita reference (BG chapter.verse) when you quote or summarise a verse.\n\n"
        "Important boundaries:\n"
        "• Do NOT give medical, legal, or financial advice.\n"
        "• Do NOT claim to be the real Krishna.\n"
        "• If the question is not directly answerable from the Bhagavad Gita, clearly say that "
        "the Bhagavad Gita does not address it, and politely invite the user to ask a Gita-based question.\n\n"
        "Your goal: leave the user feeling calmer, clearer, and gently guided by the wisdom of the Bhagavad Gita."
    )


def ask_gemini(question: str, verses: List[dict]) -> Tuple[str, List[str]]:
    context_lines = []
    refs = []
    for v in verses:
        ref = v.get("ref")  # e.g. "BG 2.47"
        text = v.get("text")
        context_lines.append(f"{ref}: {text}")
        refs.append(ref)

    context_block = "\n".join(context_lines)

    prompt = (
        build_system_prompt()
        + "\n\nRelevant verses:\n"
        + context_block
        + "\n\nUser question:\n"
        + question
        + (
            "\n\nNow respond exactly in that 3-block format. "
            "Put each sentence on a new line. "
            "Do not add anything outside those 3 parts."
        )
    )

    try:
        chat = client.chats.create(model=CHAT_MODEL)
        resp = chat.send_message(prompt)
        return resp.text, sorted(set(refs))
    except Exception:
        fallback = (
            "Right now I cannot answer because the system has reached its usage limit.\n"
            "Please try again after some time, or quietly reflect on these verses yourself."
        )
        return fallback, sorted(set(refs))


def build_reference_links(refs: List[str]) -> List[str]:
    """
    Turn refs like 'BG 11.31' into Vedabase URLs:
    https://vedabase.io/en/library/bg/11/31/
    """
    base = "https://vedabase.io/en/library/bg"
    links: List[str] = []

    for r in refs:
        cleaned = r.replace("BG", "").replace("Bg", "").strip()  # "11.31"
        parts = cleaned.split(".")
        if len(parts) == 2:
            chap, verse = parts
            chap = chap.strip()
            verse = verse.strip()
            url = f"{base}/{chap}/{verse}/"
        else:
            # fallback: chapter only
            chap = cleaned.strip()
            url = f"{base}/{chap}/"
        links.append(url)

    return links


@app.get("/health")
async def health():
    return {"status": "ok", "verses_indexed": len(META)}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if is_out_of_gita_scope(question):
        return polite_out_of_scope_reply()

    verses = search_verses(question)

    if not verses:
        return polite_out_of_scope_reply()

    answer, refs = ask_gemini(question, verses)
    links = build_reference_links(refs)

    return AskResponse(answer=answer, references=refs, reference_links=links)
from pydantic import EmailStr

class ContactRequest(BaseModel):
    name: str
    phone: str
    email: EmailStr
    subject: str
    message: str

class ContactResponse(BaseModel):
    ok: bool
    detail: str


@app.post("/contact", response_model=ContactResponse)
async def contact(req: ContactRequest):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "name": req.name,
        "phone": req.phone,
        "email": req.email,
        "subject": req.subject,
        "message": req.message,
    }

    try:
        # 1) Save to CSV
        file_exists = os.path.exists(CONTACT_CSV)
        with open(CONTACT_CSV, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print("CONTACT FORM ROW:", row)

        # 2) Try email (do not fail user if email breaks)
        try:
            send_contact_email(row)
        except Exception as e:
            print("EMAIL ERROR:", e)

        return ContactResponse(
            ok=True,
            detail="Thank you for contacting ISKCON Kurnool. We will get back to you soon."
        )
    except Exception as e:
        print("CONTACT ERROR:", e)
        raise HTTPException(status_code=500, detail="Contact save failed")
