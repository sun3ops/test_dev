"""
IT Support Knowledge Base Assistant
=====================================
A conversational AI assistant for non-technical / semi-technical employees.
Powered by: LangChain + LangGraph (ReAct) + ChromaDB + Ollama gte-large (embeddings)
LLM is fully configurable via .env (model-agnostic).

Features
--------
• PDF upload & RAG pipeline (ChromaDB + gte-large embeddings)
• Full multi-turn conversation memory via LangGraph MemorySaver
• "New Chat" button to clear conversation while keeping the KB
• Structured numbered-step answers tuned for non-technical employees
• Source attribution on every answer
• Configurable LLM via .env
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
    page_title="IT Help Desk Assistant",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ── LLM config (model-agnostic via .env) ─────────────────────────────────────
LLM_BASE_URL   = os.getenv("LLM_BASE_URL",   "https://genailab.tcs.in")
LLM_MODEL      = os.getenv("LLM_MODEL",      "azure_ai/genailab-maas-DeepSeek-R1")
LLM_API_KEY    = os.getenv("LLM_API_KEY",    "YOUR_API_KEY")
LLM_VERIFY_SSL = os.getenv("LLM_VERIFY_SSL", "false").lower() != "true"

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_kb")

if not LLM_API_KEY or LLM_API_KEY == "YOUR_API_KEY":
    st.warning("🚨 Set **LLM_API_KEY** in your `.env` file to activate the assistant.")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Custom CSS — clean enterprise helpdesk aesthetic
#     Dark navy + electric-cyan accent; IBM Plex Mono titles, Source Sans body
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Source+Sans+3:wght@300;400;600&display=swap');

/* ── global reset ── */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #0d1117;
    color: #cdd9e5;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #8b949e !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #58a6ff !important; }
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
    border-color: #58a6ff;
    color: #58a6ff !important;
    background: #0d1117;
}

/* ── main title ── */
.kb-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #58a6ff;
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

/* ── suggestion chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1rem; }
.chip {
    background: #161b22;
    border: 1px solid #30363d;
    color: #8b949e;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Source Sans 3', sans-serif;
    cursor: pointer;
    transition: all 0.18s;
    white-space: nowrap;
}
.chip:hover { border-color: #58a6ff; color: #58a6ff; background: #0d1117; }

/* ── chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-radius: 0 !important;
    padding: 0.1rem 0 !important;
}
/* user bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #161b22 !important;
}

/* ── chat input ── */
[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important;
}

/* ── status badge ── */
.status-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 12px;
    background: #1f6feb22;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    margin-left: 6px;
    vertical-align: middle;
}
.status-badge.empty {
    background: #f8514922;
    border-color: #f85149;
    color: #f85149;
}

/* ── source pill inside answer ── */
.src-pill {
    display: inline-block;
    background: #1f6feb18;
    border: 1px solid #1f6feb44;
    color: #79c0ff;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    margin: 2px 2px 0 0;
}

/* ── empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #484f58;
}
.empty-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.empty-text { font-size: 0.9rem; color: #6e7681; }

/* ── spinner override ── */
.stSpinner > div { border-top-color: #58a6ff !important; }

/* general streamlit tweaks */
.stAlert { border-radius: 6px !important; }
.stSuccess { background: #1a2b1a !important; border-color: #3fb950 !important; }
.stInfo    { background: #1a2233 !important; border-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  RAG Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to vector store…")
def get_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(
        model="gte-large",
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name="it_helpdesk_kb",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def ingest_pdf(file_bytes: bytes, filename: str, vector_store: Chroma) -> int:
    """Chunk a PDF and upsert embeddings into ChromaDB. Returns chunk count (0 = duplicate)."""
    file_id = _file_hash(file_bytes)

    existing = vector_store.get(where={"file_hash": file_id})
    if existing and existing.get("ids"):
        return 0

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
# 4.  LangGraph Tools
# ─────────────────────────────────────────────────────────────────────────────

def _vs() -> Chroma:
    return get_vector_store()


@tool
def search_it_knowledge_base(query: str, k: int = 5) -> str:
    """
    Search the IT support knowledge base for troubleshooting guides, FAQs,
    how-to steps, software access instructions, or configuration help.
    Always call this tool before answering any IT-related question.
    Returns the most relevant passages with source file and page number.
    """
    vs = _vs()
    results = vs.similarity_search_with_relevance_scores(query, k=k)

    if not results:
        return (
            "NO_RESULTS: The knowledge base has no relevant articles for this query. "
            "Tell the user politely that no matching documentation was found and suggest "
            "they contact the IT helpdesk or upload the relevant guide."
        )

    formatted = []
    for doc, score in results:
        meta = doc.metadata
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
    Returns a list of all IT knowledge base documents currently available.
    Use when the user asks 'what documents do you have?' or 'what can you help me with?'.
    """
    vs = _vs()
    result = vs.get()
    if not result or not result.get("ids"):
        return "The knowledge base is currently empty. Please upload IT support PDF documents via the sidebar."

    seen: dict[str, int] = {}
    for meta in result.get("metadatas", []):
        fname = meta.get("filename", "unknown")
        fhash = meta.get("file_hash", fname)
        if fhash not in seen:
            seen[fhash] = {"filename": fname, "chunks": 0}
        seen[fhash]["chunks"] += 1

    lines = [f"• {v['filename']} ({v['chunks']} sections)" for v in seen.values()]
    return "Available IT knowledge base documents:\n" + "\n".join(lines)


tools = [search_it_knowledge_base, list_available_documents]

# ─────────────────────────────────────────────────────────────────────────────
# 5.  LLM & Agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an IT Help Desk Assistant for employees who are not technical.
Your job is to help them solve common IT problems step by step, in plain English.

PERSONALITY
- Warm, patient, and encouraging. Never make the user feel dumb.
- Acknowledge the problem first ("That sounds frustrating — let's fix it together.")
- Use simple language. Avoid jargon unless you explain it immediately.

ANSWER FORMAT (always follow this structure)
1. One sentence acknowledging / summarising the issue.
2. Numbered steps — each step is a single clear action.
3. A short "if this doesn't work" tip or "contact IT helpdesk" fallback.
4. At the end, add a 📄 Sources line citing the document name and page number
   from the tool results (e.g. "📄 Source: VPN_Guide.pdf, Page 3").

TOOL USAGE RULES
- ALWAYS call search_it_knowledge_base before answering any IT question.
- If the user asks what help is available, call list_available_documents.
- If the tool returns NO_RESULTS, tell the user no matching article was found
  and suggest they contact IT support — do NOT make up steps.
- Never invent troubleshooting steps that aren't in the retrieved documents.

SCOPE
- Only answer IT-related questions (software, hardware, access, passwords,
  connectivity, corporate apps, etc.).
- For non-IT questions, politely redirect: "I'm an IT assistant — for that
  question you may want to reach out to [HR / Finance / your manager]."
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
# 6.  Session State
# ─────────────────────────────────────────────────────────────────────────────

def _new_thread_id() -> str:
    return str(uuid.uuid4())

if "thread_id"      not in st.session_state:
    st.session_state.thread_id      = _new_thread_id()
if "display_msgs"   not in st.session_state:
    st.session_state.display_msgs   = []   # list of {"role": "user"|"assistant", "content": str}
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🖥️ IT Help Desk")
    st.markdown("---")

    # ── New Chat ──────────────────────────────────────────────────────────────
    if st.button("💬 New Chat", use_container_width=True):
        st.session_state.thread_id    = _new_thread_id()
        st.session_state.display_msgs = []
        st.rerun()

    st.markdown("---")

    # ── KB status badge ───────────────────────────────────────────────────────
    try:
        vs     = get_vector_store()
        empty  = kb_is_empty(vs)
        badge  = "EMPTY" if empty else "ACTIVE"
        bcls   = "empty" if empty else ""
        st.markdown(
            f"**Knowledge Base** <span class='status-badge {bcls}'>{badge}</span>",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown("**Knowledge Base** <span class='status-badge empty'>OFFLINE</span>",
                    unsafe_allow_html=True)

    st.caption("Upload IT support guides, FAQs, or troubleshooting PDFs below.")

    # ── PDF uploader ──────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
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
                st.info(f"Already indexed: **{uf.name}**")
                continue
            with st.spinner(f"Indexing **{uf.name}**…"):
                n = ingest_pdf(file_bytes, uf.name, vs)
            if n == 0:
                st.info(f"Already in KB: **{uf.name}**")
            else:
                st.success(f"✅ **{uf.name}** — {n} sections indexed")
                st.session_state.ingested_files.add(fhash)

    st.markdown("---")

    # ── Indexed docs list ─────────────────────────────────────────────────────
    if st.button("📋 Show indexed documents", use_container_width=True):
        try:
            vs     = get_vector_store()
            result = vs.get()
            if result and result.get("ids"):
                seen: dict[str, int] = {}
                for meta in result.get("metadatas", []):
                    key  = meta.get("filename", "unknown")
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
                st.success("Knowledge base cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.caption(
        "**Stack**\n"
        "- LangChain + LangGraph ReAct\n"
        "- Ollama `gte-large` embeddings\n"
        "- ChromaDB (persistent)\n"
        "- MemorySaver (multi-turn memory)\n"
        f"- Model: `{LLM_MODEL}`"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main Chat UI
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='kb-title'>🖥️ IT Help Desk Assistant</div>"
    "<div class='kb-subtitle'>Ask me anything about IT — passwords, VPN, email, software access, and more.</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='kb-divider'>", unsafe_allow_html=True)

# ── Suggestion chips (only when chat is empty) ────────────────────────────────
SUGGESTIONS = [
    "How do I reset my VPN?",
    "Outlook is not syncing my emails",
    "How do I get access to Jira?",
    "I'm getting a 403 error when logging in",
    "How do I set up multi-factor authentication?",
    "My laptop won't connect to Wi-Fi",
]

if not st.session_state.display_msgs:
    st.markdown(
        "<div class='empty-state'>"
        "<div class='empty-icon'>💬</div>"
        "<div class='empty-text'>Hi! I'm your IT Help Desk assistant.<br>"
        "Ask me any IT question or tap a common issue below.</div>"
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
prompt = st.chat_input("Ask your IT question here…") or pending

if prompt:
    # Show user message immediately
    st.session_state.display_msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            try:
                agent, _ = get_agent_and_memory()

                config   = {"configurable": {"thread_id": st.session_state.thread_id}}

                # Build the message — MemorySaver holds full history internally,
                # so we only send the new user message + system prompt on first turn.
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
                err = f"❌ **Error:** {exc}\n\nCheck that Ollama is running (`ollama serve`) and your `LLM_API_KEY` is set correctly."
                st.error(err)
                st.session_state.display_msgs.append({"role": "assistant", "content": err})
