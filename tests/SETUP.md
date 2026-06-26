# IT Help Desk Assistant — Setup Guide

## What This App Does

A conversational AI assistant for non-technical employees to self-serve IT support.
Upload your company's IT guides as PDFs; employees ask questions in plain English
and receive structured, step-by-step answers grounded in your actual documentation.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the embedding model (Ollama)
```bash
# Install Ollama: https://ollama.com/download
ollama pull gte-large      # ~670 MB, one-time download
ollama serve               # runs on http://localhost:11434
```

### 3. Configure your LLM
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
# Edit .env with your LLM provider details
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Upload your PDFs
Use the sidebar to upload IT support guides, FAQs, SOPs, etc.

---

## Supported LLM Providers (configure via .env)

| Provider        | LLM_BASE_URL                        | Example Model                          |
|-----------------|--------------------------------------|----------------------------------------|
| TCS GenAI       | `https://genailab.tcs.in`           | `azure_ai/genailab-maas-DeepSeek-R1`  |
| OpenAI          | `https://api.openai.com/v1`         | `gpt-4o`                               |
| Azure OpenAI    | `https://<resource>.openai.azure.com` | `gpt-4`                              |
| Local Ollama    | `http://localhost:11434/v1`          | `llama3`                               |
| Groq            | `https://api.groq.com/openai/v1`    | `llama-3.1-70b-versatile`             |

---

## Key Features

| Feature | Detail |
|---------|--------|
| Multi-turn memory | LangGraph `MemorySaver` — full conversation context per session |
| New Chat | Clears conversation memory; KB stays intact |
| Suggestion chips | 6 common IT questions shown on empty chat |
| Structured answers | Numbered steps + acknowledgement + source citation |
| Source attribution | Every answer cites document name + page number |
| Deduplication | Re-uploading the same PDF is a no-op (SHA-256 check) |
| Persistent KB | ChromaDB persists to `./chroma_kb` across restarts |

---

## RAG Pipeline

```
PDF upload (sidebar)
      │
      ▼
PyPDFLoader  →  RecursiveCharacterTextSplitter
                 chunk_size=1000, overlap=200
      │
      ▼
OllamaEmbeddings (gte-large, local)
      │
      ▼
ChromaDB collection: "it_helpdesk_kb"
      │
      ▼  (at query time)
similarity_search_with_relevance_scores(query, k=5)
      │
      ▼
LangGraph ReAct Agent  →  LLM synthesises structured answer
```

---

## Recommended PDF Types to Upload

- VPN setup and troubleshooting guide
- Password reset and MFA instructions  
- Email client (Outlook / Gmail) configuration
- Jira, Confluence, or other tool access guides
- Onboarding IT checklist for new joiners
- Printer and peripherals setup
- Software installation guides
- Common error codes and fixes

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No relevant documents found" | Upload the relevant PDF via sidebar |
| Ollama connection error | Run `ollama serve` in a terminal |
| LLM API error | Check `LLM_API_KEY` and `LLM_BASE_URL` in `.env` |
| SSL errors | Set `LLM_VERIFY_SSL=false` in `.env` |
| Slow first response | gte-large model warming up — wait 10–15s |
