# app/v1/endpoints/ai.py
# RAG-powered AI chat with Darija support + conversation history in DB
import json
import re

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.core.errors import ERROR_RESPONSES, APIError
from app.models.chat import ChatConversation
from app.models.user import User
from app.schemas.chat import MAX_MESSAGE_LENGTH, ChatOut
from app.services.chat import ask

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
    HAS_LIMITER = True
except ImportError:
    HAS_LIMITER = False

router = APIRouter(prefix="/ai", tags=["AI Assistant"])

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"


def get_groq_key() -> str:
    key = settings.GROQ_API_KEY
    if not key:
        raise APIError(503, "llm_error", "GROQ_API_KEY not configured.")
    return key


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

DARIJA_WORDS = [
    "علاش", "كيفاش", "نجم", "باش", "وين", "فاما", "يزي", "برشا",
    "نحب", "تعمل", "نعمل", "شنية", "احنا", "هوما", "نمشي", "تمشي",
    "وقتاش", "قداش", "بالله", "تخدم", "نخدم", "لازم", "ما نجمش",
    "عندي", "عندك", "موش", "ماكش", "فيش", "عليش", "صاحبي", "خويا",
]

FRENCH_WORDS = [
    "comment", "pourquoi", "entreprise", "créer", "création",
    "quelle", "quels", "obtenir", "déclarer", "enregistrer",
    "procédure", "startup", "déclaration", "employeur", "salarié",
    "cotisation", "quoi", "peut", "dois", "faut", "besoin",
]


def detect_language(text: str) -> str:
    if re.search(r'[\u0600-\u06FF]', text):
        return "darija"
    text_lower = text.lower()
    if any(w in text_lower for w in FRENCH_WORDS):
        return "fr"
    return "en"


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS — anti-hallucination renforcé
# ══════════════════════════════════════════════════════════════════════════════

# TODO: the prompts below (down to build_rag_prompt) are unused since /chat/stream moved to RAGPipeline.
# Remove them once Wiem decides on Darija support.
STRICT_GROUNDING = """قواعد أساسية / Règles fondamentales:
1. أجب فقط بناءً على المعلومات الموجودة في الوثائق أدناه
2. لا تخترع معلومات / N'invente aucune information
3. لا تستخدم معرفتك العامة / N'utilise pas tes connaissances générales
4. كل جملة لازم تكون مأخوذة من الوثائق — ممنوع الإضافة
5. Si une information n'est pas dans les documents, ne la mentionne PAS"""

SYSTEM_PROMPTS = {
    "darija": """أنت مساعد إداري تونسي متخصص.
قواعد صارمة جداً:
- اكتب بالدارجة التونسية فقط — مش مصرية، مش فصحى، مش فرنسية
- الدارجة التونسية تستعمل: لازم، نجم، تمشي، تجيب، باش، كيفاش، برشا، موش، وقتاش، قداش، شنية، عندي، ماكش
- المصطلحات الرسمية تبقى كما هي: RNE، CNSS، SARL، SA، API
- ممنوع تكتب كلمة واحدة بالفرنسية أو العربية الفصحى أو المصرية
- ممنوع تخترع معلومات — اكتب فقط ما هو موجود في الوثائق
- إذا ما عندكش المعلومة في الوثائق قل: "هاذي المعلومة موش موجودة في الوثائق"
- ممنوع تقول "يمكن" أو "ربما" — كل جملة لازم تكون من الوثائق فقط""",

    "fr": """Tu es un assistant administratif tunisien spécialisé.
Règles strictes:
- Écris UNIQUEMENT en français standard
- Les termes officiels restent tels quels: RNE, CNSS, SARL, SA, API
- N'invente aucune information — réponds uniquement sur base des documents
- Si l'info n'est pas dans les documents, dis-le clairement""",

    "en": """You are a specialized Tunisian administrative assistant.
Strict rules:
- Write ONLY in English
- Keep official terms as-is: RNE, CNSS, SARL, SA, API
- Do NOT invent information — answer only based on the documents provided
- If the info is not in the documents, say so clearly""",
}

FEW_SHOT = {
    "darija": """
=== أمثلة على الدارجة التونسية الصحيحة ===
مثال 1:
السياق: "Pour créer une entreprise: 1) Rédiger les statuts. 2) Ouvrir un compte bancaire. 3) S'immatriculer au RNE \
dans les 60 jours."
السؤال: كيفاش نعمل شركة؟
الجواب الصحيح بالدارجة: باش تعمل شركة، لازم:
1) تكتب عقد التأسيس
2) تفتح حساب بنكي باسم الشركة
3) تسجل في RNE في ظرف 60 يوم

مثال 2:
السياق: "La déclaration à la CNSS est obligatoire pour tout employeur dès le premier recrutement."
السؤال: وقتاش لازم نصرح في الكناس؟
الجواب الصحيح بالدارجة: حسب الوثائق، لازم تصرح في الكناس من أول ما تعمل الشركة وقبل ما تخدم أي واحد.
=== نهاية الأمثلة ===
""",
    "fr": """
=== Exemples ===
Contexte: "Pour créer une entreprise: rédiger les statuts, ouvrir un compte, s'immatriculer au RNE."
Question: Comment créer une entreprise?
Réponse: Selon les documents: 1) Rédiger les statuts. 2) Ouvrir un compte bancaire. 3) S'immatriculer au RNE.
=== Fin ===
""",
    "en": """
=== Examples ===
Context: "To create a company: draft statutes, open a bank account, register at the RNE."
Question: How to create a company?
Answer: According to the documents: 1) Draft the statutes. 2) Open a bank account. 3) Register at the RNE.
=== End ===
""",
}

LANG_REMINDER = {
    "darija": "\nتذكير مهم: اكتب جوابك بالدارجة التونسية فقط — مش مصرية، مش فرنسية، مش فصحى. لا تخترع معلومات.\n",
    "fr":     "\nRappel: écris UNIQUEMENT en français. N'invente aucune information.\n",
    "en":     "\nReminder: write ONLY in English. Do not invent information.\n",
}

FALLBACK = {
    "darija": "المعلومة هاذي موش موجودة في الوثائق المتوفرة.",
    "fr":     "Cette information n'est pas disponible dans les documents fournis.",
    "en":     "This information is not available in the provided documents.",
}

PUBLIC_PROMPT = """You are the LegalEase Tunisia welcome assistant.
Explain what LegalEase Tunisia is and how to use it to visitors who haven't signed up yet.
Rules:
- Respond in the SAME language as the visitor
- Keep answers SHORT — 2 to 4 sentences maximum
- Always encourage the visitor to create a free account"""


def build_rag_prompt(lang: str, context: str, question: str) -> str:
    system   = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    examples = FEW_SHOT.get(lang, FEW_SHOT["en"])
    reminder = LANG_REMINDER.get(lang, LANG_REMINDER["en"])
    fallback = FALLBACK.get(lang, FALLBACK["en"])

    if lang == "darija":
        fallback_instr = f'إذا الجواب موش موجود في السياق، قل فقط: "{fallback}"'
        q_label, a_label = "السؤال بالدارجة", "الجواب بالدارجة التونسية فقط"
    elif lang == "fr":
        fallback_instr = f'Si la réponse n\'est pas dans le contexte, dis: "{fallback}"'
        q_label, a_label = "Question", "Réponse en français uniquement"
    else:
        fallback_instr = f'If the answer is not in the context, say: "{fallback}"'
        q_label, a_label = "Question", "Answer in English only"

    return f"""{system}

{STRICT_GROUNDING}

{fallback_instr}

{examples}

=== الوثائق الرسمية / Documents officiels ===
{context}
=== نهاية الوثائق ===
{reminder}
{q_label}: {question}

{a_label}:"""


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ChatMessageSchema(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message:         str
    history:         list[ChatMessageSchema] | None = []
    lang:            str | None = "auto"
    conversation_id: int | None = None

class StreamChatRequest(ChatRequest):  # /chat/public keeps ChatRequest, without the limit
    message: str = Field(max_length=MAX_MESSAGE_LENGTH)

class ConversationOut(BaseModel):
    id:         int
    title:      str | None
    lang:       str | None
    created_at: str
    messages:   list[dict]

    class Config:
        from_attributes = True


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING HELPER
# ══════════════════════════════════════════════════════════════════════════════

async def stream_groq(messages: list[dict], api_key: str):
    async with httpx.AsyncClient(timeout=90.0) as client:
        async with client.stream(
            "POST", GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model":       GROQ_MODEL,
                "messages":    messages,
                "max_tokens":  2048,
                "temperature": 0.0,   # 0.0 = déterministe, élimine hallucinations
                "stream":      True,
            },
        ) as resp:
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': 'Groq API error'})}\n\n"
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return
                try:
                    delta = json.loads(raw)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield f"data: {json.dumps({'chunk': delta})}\n\n"
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/chat/public")
async def chat_public(request: Request, req: ChatRequest):
    """No auth — Home page welcome bot."""
    messages = [{"role": "system", "content": PUBLIC_PROMPT}]
    for m in (req.history or [])[-6:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    return StreamingResponse(
        stream_groq(messages, get_groq_key()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def answer_events(result: ChatOut):
    """The events the chat widget reads: text chunks, sources, conversation id, then [DONE] (it must be last)."""
    # Several small events: the widget does not reassemble a line split across two network reads.
    for start in range(0, len(result.answer), 200):
        yield sse({"chunk": result.answer[start : start + 200]})
    sources = {
        (c.doc_title, c.page): {"title": f"{c.doc_title}, p. {c.page}", "source": c.doc_title, "page": c.page}
        for c in result.citations
    }
    if sources:
        yield sse({"sources": list(sources.values())})
    yield sse({"conversation_id": result.conversation_id})
    yield "data: [DONE]\n\n"


# A plain `def`: the pipeline blocks, so FastAPI must run this in its thread pool.
@router.post("/chat/stream", responses=ERROR_RESPONSES, summary="Ask the legal assistant (SSE)")
def chat_stream(
    req: StreamChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Same as `POST /api/v1/chat`, as server-sent events for the chat widget.

    The pipeline does not stream, so the events are sent once the whole answer is ready.
    `history` is accepted but not used.
    """
    lang = req.lang if req.lang and req.lang != "auto" else detect_language(req.message)
    # Runs before the response starts, so failures are ordinary JSON errors.
    result = ask(db, current_user.id, req.message, req.conversation_id, lang)

    return StreamingResponse(
        answer_events(result),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/conversations")
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    convs = (
        db.query(ChatConversation)
        .filter(ChatConversation.user_id == current_user.id)
        .order_by(ChatConversation.updated_at.desc())
        .limit(20)
        .all()
    )
    return [
        {
            "id":            c.id,
            "title":         c.title or "Untitled",
            "lang":          c.lang,
            "created_at":    c.created_at.isoformat() if c.created_at else "",
            "message_count": len(c.messages),
        }
        for c in convs
    ]


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = db.query(ChatConversation).filter(
        ChatConversation.id == conversation_id,
        ChatConversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id":         conv.id,
        "title":      conv.title or "Untitled",
        "lang":       conv.lang,
        "created_at": conv.created_at.isoformat() if conv.created_at else "",
        "messages": [
            {
                "role":       m.role.value,
                "content":    m.content,
                "status":     m.status,
                "error_code": m.error_code,
                "citations":  m.citations or [],
                "created_at": m.created_at.isoformat() if m.created_at else "",
            }
            for m in conv.messages
        ],
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = db.query(ChatConversation).filter(
        ChatConversation.id == conversation_id,
        ChatConversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
    return {"message": "Conversation deleted"}

