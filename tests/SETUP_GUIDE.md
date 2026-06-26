# BA Requirements Clarification Chatbot — Setup & Run Guide

## What Changed from the Baseline

| Area | Baseline (IT Help Desk) | This Version (BA Chatbot) |
|---|---|---|
| Domain | IT troubleshooting | Business Analysis / Requirements Engineering |
| System prompt | IT support steps | Ambiguity detection + SMART requirement refinement |
| Tools | `search_it_knowledge_base` | `search_requirements_knowledge_base` |
| ChromaDB collection | `it_helpdesk_kb` | `ba_requirements_kb` |
| Persist dir | `./chroma_kb` | `./chroma_ba_kb` |
| Auto-ingest | ❌ | ✅ via `AUTO_INGEST_PDF` in `.env` |
| Suggestion chips | IT problems | Ambiguous requirement samples |
| Embedding strategy | Every session upload | **One-time** — hash-guarded, reused from disk |

---

## Prerequisites

Make sure the following are installed and running before you start.

### 1. Python environment
```bash
pip install streamlit langchain langchain-openai langchain-ollama \
            langchain-chroma langchain-community langgraph \
            chromadb pypdf python-dotenv httpx
```

### 2. Ollama — for gte-large embeddings
Ollama should already be running. Pull the embedding model if you haven't:
```bash
ollama pull gte-large
```
Verify it's running:
```bash
ollama list        # should show gte-large
ollama serve       # starts the server (if not already running)
```

---

## Step 1: Set up your `.env` file

Copy the example file and fill in your API key:
```bash
cp .env.example .env
```

Open `.env` and set:
```
LLM_API_KEY=<your actual API key here>
```

Leave everything else as-is unless your setup differs (e.g. Ollama on a different port).

---

## Step 2: Place the Knowledge Base PDF

Put the **Requirements Clarification Knowledge Base PDF** in the same folder as `app.py` and name it `requirements_kb.pdf`:

```
your-project/
├── app.py
├── .env
├── requirements_kb.pdf    ← the PDF you have
└── chroma_ba_kb/          ← auto-created by ChromaDB on first run
```

The `.env` already points to this path:
```
AUTO_INGEST_PDF=./requirements_kb.pdf
```

---

## Step 3: Run the app

```bash
streamlit run app.py
```

**What happens on first run:**
1. Streamlit starts, Ollama embeddings connect.
2. The PDF is detected via `AUTO_INGEST_PDF`, chunked, and embedded using `gte-large`.
3. Embeddings are saved to `./chroma_ba_kb/` on disk.
4. The chatbot is ready.

**What happens on every subsequent run:**
- ChromaDB detects existing embeddings in `./chroma_ba_kb/`.
- The duplicate-hash guard skips re-embedding.
- The app starts instantly — **no re-embedding cost**.

---

## Step 4: Alternatively — upload via sidebar

If you prefer not to use `AUTO_INGEST_PDF`, leave it blank in `.env` and:
1. Open the sidebar in the running app.
2. Click **Upload PDF knowledge base**.
3. Select your PDF.
4. Wait for the "✅ N sections embedded & saved to disk" message.
5. From that point on, future restarts will reuse the saved embeddings.

---

## How the one-time embedding works

```
First run:
  PDF → PyPDFLoader → RecursiveTextSplitter (1000 chars, 200 overlap)
      → OllamaEmbeddings (gte-large) → ChromaDB (saved to ./chroma_ba_kb/)

All subsequent runs:
  ChromaDB loads from ./chroma_ba_kb/ directly
  File hash check → already indexed → skip  ✅
```

The key is the **SHA-256 file hash** stored as metadata in every chunk. On upload or auto-ingest, the app queries ChromaDB for that hash. If found, it returns `0` (skipped). No Ollama call is made.

---

## Demo flow for the hackathon

1. Open the app → you'll see 6 sample ambiguous requirement chips.
2. Click **"The system should be fast and user-friendly"** — the chatbot will:
   - Detect "fast" and "user-friendly" as ambiguous
   - Ask 1–2 clarifying questions
3. Answer the questions → the chatbot refines the requirement into SMART format.
4. Click **"Managers need reports as soon as possible"** — demonstrates time-constraint ambiguity.
5. At any point, ask: **"Summarise the refined requirements so far"** — the chatbot produces a structured summary block.
6. Click **"New Chat"** to reset the conversation (KB stays intact).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` on Ollama | Run `ollama serve` in a terminal |
| `gte-large not found` | Run `ollama pull gte-large` |
| `LLM_API_KEY` warning | Set your key in `.env` |
| Embeddings seem slow on first run | Normal — gte-large is computing vectors. Subsequent runs are instant. |
| Want to re-embed the PDF | Delete `./chroma_ba_kb/` folder and restart the app |
| ChromaDB errors | `pip install --upgrade chromadb langchain-chroma` |
