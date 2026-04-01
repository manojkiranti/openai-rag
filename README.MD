# RAG Retrieval and Answer Generation Backend

FastAPI backend for retrieval and answer generation only.

This project assumes:

- your documents are already embedded
- vectors are already stored in Qdrant
- no ingestion, parsing, chunking, embedding generation, or Preeti conversion is needed here
- query embeddings are generated with the same model configured in `.env`
- answer generation uses Ollama with Qwen

---

## Features

- `GET /health` health check for Qdrant and Ollama
- `POST /ask` embed query, retrieve top-k chunks from Qdrant, send grounded prompt to Ollama, return answer + sources
- `GET /search` semantic search only, returns retrieved chunks

---

uvicorn app.main:app --reload

## Project Structure

```text
app/
├── api/routes/
│   ├── chat.py
│   └── health.py
├── db/
│   └── qdrant.py
├── models/
│   └── schemas.py
├── services/
│   ├── llm.py
│   └── retrieval.py
├── config.py
└── main.py