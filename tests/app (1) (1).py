"""
Business Analysis Requirements Clarification Chatbot
=====================================================
AI-powered chatbot that converses with stakeholders to clarify and refine
business requirements — detects ambiguity, asks context-aware questions,
manages multi-turn dialogue, and summarises refined requirements.

Stack: LangChain + LangGraph (ReAct) + ChromaDB + Ollama gte-large (embeddings)
LLM is fully configurable via .env (model-agnostic).

Features
--------
• One-time PDF ingestion → ChromaDB (embeddings persist on disk; re-used forever)
• Full multi-turn conversation memory via LangGraph MemorySaver
• Ambiguity detection & context-aware clarification questions
• "New Chat" button to reset conversation (KB stays intact)
• Refined requirement summary on demand
• Source attribution on every answer
"""

import os
import tempfile
import hashlib
import uuid
import streamlit as st
import httpx
from dotenv import load_dotenv

# ── LangChain / LangGraph ─────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Page Config & Environment
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BA Requirements Clarifier",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ── LLM config (model-agnostic via .env) ─────────────────────────────────────
LLM_BASE_URL   = os.getenv("LLM_BASE_URL",   "https://genailab.tcs.in")
LLM_MODEL      = os.getenv("LLM_MODEL",      "azure_ai/genailab-maas-DeepSeek-R1")
LLM_API_KEY    = os.getenv("LLM_API_KEY",    "YOUR_API_KEY")
LLM_VERIFY_SSL = os.getenv("LLM_VERIFY_SSL", "false").lower() != "true"

OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_ba_kb")

# ── Auto-ingest PDF path (set this if you want one-time ingestion on startup) ─
AUTO_INGEST_PDF = os.getenv("AUTO_INGEST_PDF", "")   # e.g. "./requirements_kb.pdf"

if not LLM_API_KEY or LLM_API_KEY == "YOUR_API_KEY":
    st.warning("🚨 Set **LLM_API_KEY** in your `.env` file to activate the assistant.")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Custom CSS — clean professional BA tool aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #0d1117;
    color: #cdd9e5;
}

section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #8b949e !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #d2a679 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9 !important;
    border-radius: 6px;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.85rem;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #d2a679;
    color: #d2a679 !important;
    background: #0d1117;
}

.kb-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #d2a679;
    letter-spacing: -0.03em;
    margin-bottom: 0;
}
.kb-subtitle {
    font-size: 0.9rem;
    color: #8b949e;
    margin-top: 2px;
    margin-bottom: 1rem;
}
.kb-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 0.5rem 0 1.2rem 0;
}

.status-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 12px;
    background: #d2a67922;
    border: 1px solid #d2a679;
    color: #d2a679;
    margin-left: 6px;
    vertical-align: middle;
}
.status-badge.empty {
    background: #f8514922;
    border-color: #f85149;
    color: #f85149;
}

.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #484f58;
}
.empty-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.empty-text { font-size: 0.9rem; color: #6e7681; }

[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #d2a679 !important;
    box-shadow: 0 0 0 3px rgba(210,166,121,0.12) !important;
}

.stSpinner > div { border-top-color: #d2a679 !important; }
.stAlert { border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  RAG Infrastructure  (ONE-TIME embedding, then reuse from disk)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to vector store…")
def get_vector_store() -> Chroma:
    """
    Returns a ChromaDB collection backed by gte-large embeddings.
    If the persist directory already has data, no new embeddings are computed
    — ChromaDB loads them directly from disk.
    """
    embeddings = OllamaEmbeddings(
        model="gte-large",
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name="ba_requirements_kb",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def ingest_pdf(file_bytes: bytes, filename: str, vector_store: Chroma) -> int:
    """
    Chunk a PDF and upsert embeddings into ChromaDB.
    Returns number of chunks added (0 = already indexed — skipped).
    This function is idempotent: calling it twice on the same file is safe.
    """
    file_id = _file_hash(file_bytes)

    # ── Duplicate guard: skip if this file hash already exists in the KB ──────
    existing = vector_store.get(where={"file_hash": file_id})
    if existing and existing.get("ids"):
        return 0   # already embedded — reuse from disk

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        raw_pages: list[Document] = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = splitter.split_documents(raw_pages)

    ids, texts, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{file_id}_{i}")
        texts.append(chunk.page_content)
        metadatas.append({
            **chunk.metadata,
            "filename": filename,
            "file_hash": file_id,
            "chunk_index": i,
        })

    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(chunks)


def kb_is_empty(vector_store: Chroma) -> bool:
    result = vector_store.get()
    return not (result and result.get("ids"))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Auto-ingest from AUTO_INGEST_PDF env var (one-time on startup)
# ─────────────────────────────────────────────────────────────────────────────

def _auto_ingest_if_needed():
    """
    If AUTO_INGEST_PDF is set and the file exists, ingest it once.
    Subsequent runs skip it because of the file-hash duplicate guard.
    """
    if not AUTO_INGEST_PDF:
        return
    pdf_path = os.path.abspath(AUTO_INGEST_PDF)
    if not os.path.isfile(pdf_path):
        st.sidebar.warning(f"AUTO_INGEST_PDF not found: {pdf_path}")
        return
    vs = get_vector_store()
    with open(pdf_path, "rb") as f:
        data = f.read()
    n = ingest_pdf(data, os.path.basename(pdf_path), vs)
    if n > 0:
        st.sidebar.success(f"✅ Auto-indexed **{os.path.basename(pdf_path)}** — {n} sections")
    # If n == 0, it was already indexed on a previous run — silently skip.


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LangGraph Tools
# ─────────────────────────────────────────────────────────────────────────────

def _vs() -> Chroma:
    return get_vector_store()


@tool
def search_requirements_knowledge_base(query: str, k: int = 5) -> str:
    """
    Search the Business Analysis knowledge base for:
    - Quality frameworks (SMART, INVEST)
    - Ambiguity heuristics and linguistic red-flag patterns
    - Poor-vs-excellent requirement mappings by domain
    - Clarification question templates
    - JSON metadata structures for requirement scenarios

    Always call this tool before responding to any requirement-related question
    or when analysing stakeholder input for ambiguity.
    Returns the most relevant passages with source file and page number.
    """
    vs = _vs()
    results = vs.similarity_search_with_relevance_scores(query, k=k)

    if not results:
        return (
            "NO_RESULTS: The knowledge base has no relevant content for this query. "
            "Proceed with your built-in knowledge of requirements quality frameworks."
        )

    formatted = []
    for doc, score in results:
        meta   = doc.metadata
        source = meta.get("filename", meta.get("source", "unknown"))
        page   = meta.get("page", "?")
        formatted.append(
            f"[Relevance: {score:.2f} | Document: {source} | Page: {page}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(formatted)


@tool
def list_available_documents() -> str:
    """
    Returns a list of all knowledge base documents currently available.
    Use when the user asks 'what documents do you have?' or 'what topics can you help with?'.
    """
    vs = _vs()
    result = vs.get()
    if not result or not result.get("ids"):
        return "The knowledge base is currently empty. Please upload a requirements PDF via the sidebar."

    seen: dict[str, dict] = {}
    for meta in result.get("metadatas", []):
        fname = meta.get("filename", "unknown")
        fhash = meta.get("file_hash", fname)
        if fhash not in seen:
            seen[fhash] = {"filename": fname, "chunks": 0}
        seen[fhash]["chunks"] += 1

    lines = [f"• {v['filename']} ({v['chunks']} sections)" for v in seen.values()]
    return "Available knowledge base documents:\n" + "\n".join(lines)


tools = [search_requirements_knowledge_base, list_available_documents]

# ─────────────────────────────────────────────────────────────────────────────
# 6.  LLM, Agent & System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Business Analysis Requirements Clarification Assistant.
Your role is to engage stakeholders in natural language to clarify and refine
ambiguous or incomplete business requirement specifications.

━━━ CORE MISSION ━━━
1. DETECT ambiguity, vagueness, and incompleteness in stakeholder requirements.
2. ASK targeted, context-aware clarifying questions (one or two at a time — never overwhelming).
3. GUIDE the stakeholder through multi-turn dialogue to progressively refine each requirement.
4. SUMMARISE the refined, structured requirement at the end in SMART format.

━━━ AMBIGUITY PATTERNS TO INTERCEPT ━━━
Always check stakeholder input against these red flags and challenge them:
• Unquantified performance terms: "fast", "quickly", "scalable", "optimised", "high-throughput"
  → Ask for explicit ms/s thresholds, throughput rates, or capacity bounds.
• Subjective UX language: "user-friendly", "intuitive", "modern", "seamless"
  → Ask for specific click-path counts, accessibility standards (e.g. WCAG 2.1 AA), or UI patterns.
• Vague actors: "users", "managers", "staff", "the team"
  → Ask for the exact role name, system group, or persona.
• Unbounded lists: "etc.", "and more", "such as", "including but not limited to"
  → Ask the stakeholder to enumerate all items explicitly.
• Missing time constraints: "as soon as possible", "immediately", "regularly"
  → Demand exact schedules, latency windows, or SLA deadlines.
• Missing security/compliance context: "secure", "locked down", "safe"
  → Ask for compliance mandates (GDPR, HIPAA, PCI-DSS) and auth mechanisms.
• Missing error/exception handling: happy-path only statements
  → Ask what should happen on failure, timeout, or invalid input.

━━━ CONVERSATION STYLE ━━━
- Warm and professional. Never make the stakeholder feel criticised.
- Acknowledge what they said before probing: "That's a great start. To make this testable, I need to clarify a couple of things…"
- Ask ONE or TWO focused questions per turn. Never fire a list of 10 questions at once.
- Keep questions specific and binary where possible ("Should it be 200ms or 500ms?" is better than "How fast?").

━━━ ANSWER FORMAT ━━━
When analysing requirements:
1. **Ambiguity Detected:** List the vague phrases you found.
2. **Clarifying Questions:** Ask 1–2 targeted questions.
3. *(After stakeholder responds)* **Refined Requirement:** Rewrite the requirement in SMART language.

When asked to summarise at the end:
- Produce a structured requirement block labelled with: Actor | Trigger | System Behaviour | Performance Threshold | Security/Compliance | Error Handling.

━━━ TOOL USAGE ━━━
- ALWAYS call search_requirements_knowledge_base before analysing any requirement or answering a question about BA frameworks.
- Use retrieved SMART/INVEST frameworks and poor-vs-excellent scenario mappings to guide your clarifications.
- If the tool returns NO_RESULTS, fall back to your built-in knowledge.
- At the end of each answer, cite the source: 📄 Source: [filename], Page [n].

━━━ SCOPE ━━━
- Only handle business analysis, requirements engineering, and project scoping topics.
- For unrelated questions, politely redirect: "I'm a requirements clarification assistant — for that topic you may want to reach out to the relevant team."
"""


@st.cache_resource(show_spinner="Initialising assistant…")
def get_agent_and_memory():
    http_client = httpx.Client(verify=not LLM_VERIFY_SSL)
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        http_client=http_client,
    )
    memory = MemorySaver()
    agent  = create_react_agent(llm, tools, checkpointer=memory)
    return agent, memory


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Session State
# ─────────────────────────────────────────────────────────────────────────────

def _new_thread_id() -> str:
    return str(uuid.uuid4())

if "thread_id"      not in st.session_state:
    st.session_state.thread_id      = _new_thread_id()
if "display_msgs"   not in st.session_state:
    st.session_state.display_msgs   = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Auto-ingest on startup (runs once; duplicate guard prevents re-embedding)
# ─────────────────────────────────────────────────────────────────────────────
_auto_ingest_if_needed()

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📋 BA Requirements Clarifier")
    st.markdown("---")

    if st.button("💬 New Chat", use_container_width=True):
        st.session_state.thread_id    = _new_thread_id()
        st.session_state.display_msgs = []
        st.rerun()

    st.markdown("---")

    # ── KB status ─────────────────────────────────────────────────────────────
    try:
        vs    = get_vector_store()
        empty = kb_is_empty(vs)
        badge = "EMPTY" if empty else "ACTIVE"
        bcls  = "empty" if empty else ""
        st.markdown(
            f"**Knowledge Base** <span class='status-badge {bcls}'>{badge}</span>",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown("**Knowledge Base** <span class='status-badge empty'>OFFLINE</span>",
                    unsafe_allow_html=True)

    st.caption(
        "Upload the Requirements Clarification Knowledge Base PDF below.\n"
        "Embeddings are computed **once** and reused from disk on every restart."
    )

    # ── PDF uploader ──────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDF knowledge base",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        vs = get_vector_store()
        for uf in uploaded_files:
            file_bytes = uf.read()
            fhash      = _file_hash(file_bytes)
            if fhash in st.session_state.ingested_files:
                st.info(f"Already indexed this session: **{uf.name}**")
                continue
            with st.spinner(f"Embedding **{uf.name}** (one-time)…"):
                n = ingest_pdf(file_bytes, uf.name, vs)
            if n == 0:
                st.info(f"Already in KB (loaded from disk): **{uf.name}**")
            else:
                st.success(f"✅ **{uf.name}** — {n} sections embedded & saved to disk")
                st.session_state.ingested_files.add(fhash)

    st.markdown("---")

    if st.button("📋 Show indexed documents", use_container_width=True):
        try:
            vs     = get_vector_store()
            result = vs.get()
            if result and result.get("ids"):
                seen: dict[str, int] = {}
                for meta in result.get("metadatas", []):
                    key       = meta.get("filename", "unknown")
                    seen[key] = seen.get(key, 0) + 1
                for fname, cnt in seen.items():
                    st.markdown(f"- **{fname}** ({cnt} sections)")
            else:
                st.warning("No documents indexed yet.")
        except Exception as e:
            st.error(f"Could not read KB: {e}")

    st.markdown("---")

    with st.expander("⚠️ Admin"):
        if st.button("🗑 Clear knowledge base", use_container_width=True):
            try:
                vs = get_vector_store()
                vs.delete_collection()
                st.session_state.ingested_files.clear()
                st.cache_resource.clear()
                st.success("Knowledge base cleared. Refresh the page.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.caption(
        "**Stack**\n"
        "- LangChain + LangGraph ReAct\n"
        "- Ollama `gte-large` embeddings\n"
        "- ChromaDB (persistent — one-time embedding)\n"
        "- MemorySaver (multi-turn memory)\n"
        f"- Model: `{LLM_MODEL}`"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Main Chat UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<div class='kb-title'>📋 BA Requirements Clarification Assistant</div>"
    "<div class='kb-subtitle'>Paste your requirement statement and I'll detect ambiguities, "
    "ask clarifying questions, and help you refine it into a SMART specification.</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='kb-divider'>", unsafe_allow_html=True)

# ── Suggestion chips (only when chat is empty) ────────────────────────────────
SUGGESTIONS = [
    "The system should be fast and user-friendly",
    "Managers need reports as soon as possible",
    "Only authorised users can see sensitive data",
    "The app should handle a lot of users at once",
    "Explain the SMART framework for requirements",
    "What ambiguity patterns should I watch for?",
]

if not st.session_state.display_msgs:
    st.markdown(
        "<div class='empty-state'>"
        "<div class='empty-icon'>📋</div>"
        "<div class='empty-text'>Hi! I'm your Business Analysis Requirements Clarification Assistant.<br>"
        "Paste a requirement statement or tap a sample below to get started.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for i, sug in enumerate(SUGGESTIONS):
        if cols[i % 3].button(sug, key=f"chip_{i}", use_container_width=True):
            st.session_state["_pending_prompt"] = sug
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.display_msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle pending chip selection ─────────────────────────────────────────────
pending = st.session_state.pop("_pending_prompt", None)

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Paste a requirement statement or ask a BA question…") or pending

if prompt:
    st.session_state.display_msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing requirements…"):
            try:
                agent, _ = get_agent_and_memory()
                config   = {"configurable": {"thread_id": st.session_state.thread_id}}

                events = agent.stream(
                    {
                        "messages": [
                            SystemMessage(content=SYSTEM_PROMPT),
                            HumanMessage(content=prompt),
                        ]
                    },
                    config=config,
                    stream_mode="values",
                )

                final_messages = None
                for event in events:
                    final_messages = event["messages"]

                answer = ""
                if final_messages:
                    last = final_messages[-1]
                    if isinstance(last, AIMessage) and last.content:
                        answer = last.content

                if answer:
                    st.markdown(answer)
                    st.session_state.display_msgs.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    fallback = "I couldn't generate a response. Please try again."
                    st.markdown(fallback)
                    st.session_state.display_msgs.append(
                        {"role": "assistant", "content": fallback}
                    )

            except Exception as exc:
                err = (
                    f"❌ **Error:** {exc}\n\n"
                    "Check that:\n"
                    "1. Ollama is running (`ollama serve`)\n"
                    "2. `LLM_API_KEY` is set correctly in `.env`\n"
                    "3. The `gte-large` model is pulled (`ollama pull gte-large`)"
                )
                st.error(err)
                st.session_state.display_msgs.append(
                    {"role": "assistant", "content": err}
                )
