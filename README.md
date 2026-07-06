# Interview Trainer Agent — IBM Granite (PS #22)

An AI-powered interview preparation assistant built with **IBM watsonx.ai (Granite-13b-chat-v2)** and **RAG (Retrieval-Augmented Generation)**.

## Architecture

```
interview-trainer-agent/
├── backend/
│   ├── app.py              ← Flask REST API (4 endpoints)
│   ├── rag_engine.py       ← FAISS vector index + semantic retrieval
│   ├── prompt_builder.py   ← Structured Granite prompt templates
│   ├── watsonx_client.py   ← IBM watsonx.ai Granite integration
│   ├── requirements.txt
│   └── .env.example        ← Copy to .env and fill credentials
├── data/
│   └── interview_corpus.json  ← 22-entry curated Q&A knowledge base
└── frontend/
    └── index.html          ← Self-contained chat UI (no build step)
```

## Features

| Feature | Description |
|---|---|
| **Generate Questions** | Enter role + experience + topic → get 5 tailored questions with model answers and tips |
| **Answer Feedback** | Paste your answer → get score, strengths, improvements, and a revised answer |
| **Chat with Agent** | Free-form conversation with the interview coach |
| **RAG** | FAISS semantic search over the curated knowledge base for grounded responses |
| **IBM Granite** | Uses `ibm/granite-13b-chat-v2` via watsonx.ai Lite tier |

## Quick Start

### 1. Install dependencies

```bash
cd interview-trainer-agent/backend
pip install -r requirements.txt
```

### 2. Set IBM Credentials

```bash
copy .env.example .env
# Edit .env and fill in WATSONX_API_KEY and WATSONX_PROJECT_ID
```

**How to get IBM credentials (free):**
1. Sign up at https://cloud.ibm.com (Lite tier — no credit card)
2. Create a "Watson Machine Learning" service instance
3. Go to https://dataplatform.cloud.ibm.com → create a project
4. Copy the Project ID from project settings
5. Go to IBM Cloud → Manage → Access (IAM) → API Keys → Create

### 3. Start the backend

```bash
cd interview-trainer-agent/backend
python app.py
```

The API starts at `http://localhost:5000`.

### 4. Open the frontend

Open `interview-trainer-agent/frontend/index.html` in any browser. No build step needed.

> **Demo mode**: If IBM credentials are not set, the app runs in demo mode with sample responses.

## API Reference

### `POST /api/generate`
```json
{
  "role": "Software Engineer",
  "experience": "Junior (1–3 years)",
  "focus_area": "system design and algorithms"
}
```

### `POST /api/feedback`
```json
{
  "role": "Data Scientist",
  "question": "What is the bias-variance trade-off?",
  "answer": "It is about balancing how well the model fits data..."
}
```

### `POST /api/chat`
```json
{ "message": "What are the most common questions for React interviews?" }
```

### `GET /api/health`
```json
{ "status": "ok", "service": "Interview Trainer Agent" }
```

## IBM Tech Stack (Lite Tier)

| Component | IBM Service |
|---|---|
| LLM | watsonx.ai — `ibm/granite-13b-chat-v2` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Search | FAISS (local, free) |
| Backend | Flask (Python) |
| Storage | Local JSON corpus (expandable to IBM COS) |

## Extending the Knowledge Base

Add entries to `data/interview_corpus.json` following the schema:
```json
{
  "role": "Your Role",
  "category": "Technical | Behavioral | HR | System Design",
  "question": "...",
  "model_answer": "...",
  "tips": "..."
}
```

The RAG index rebuilds automatically on the next server start.

---

*IBM-Edunet AICTE Internship · Problem Statement #22*
