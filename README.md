# Multi-Document Hospital RAG Assistant

A production-style Retrieval-Augmented Generation system for hospital patient queries. The app ingests one or more hospital PDFs, stores chunk embeddings in Supabase with `pgvector`, and answers questions using only retrieved document context.

## Architecture
```text
        +--------------------+
        |   Streamlit UI     |
        | upload + chat      |
        +---------+----------+
                  |
                  v
        +--------------------+
        |   FastAPI Backend  |
        | upload/query APIs  |
        +----+----------+----+
             |          |
             |          v
             |    +-----------+
             |    |   Groq    |
             |    |   LLM     |
             |    +-----------+
             v
    +------------------------+
    |  Local Embeddings      |
    | BAAI/bge-small-en-v1.5 |
    +-----------+------------+
                |
                v
    +------------------------+
    | Supabase + pgvector    |
    | docs, chunks, hybrid   |
    +------------------------+
```

## Features
- Multi-document PDF ingestion
- Supabase-backed vector storage with `pgvector`
- Hybrid retrieval using embeddings plus PostgreSQL full-text search
- Strict grounded answering with exact fallback behavior
- Document-aware citations and source chunk display
- Streamlit dark-mode chat UI with upload, scope filtering, and document removal

## Project Structure
```text
project/
|-- backend/
|   |-- main.py
|   |-- ingest.py
|   |-- rag_pipeline.py
|   |-- db.py
|   `-- config.py
|-- frontend/
|   `-- app.py
|-- utils/
|   |-- chunking.py
|   `-- embedding.py
|-- sql/
|   `-- schema.sql
|-- tests/
|-- requirements.txt
|-- .env.example
`-- README.md
```

## Environment Setup
1. Create and activate a virtual environment.
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
2. Install dependencies.
```powershell
pip install -r requirements.txt
```
3. Create `.env` from `.env.example` and fill in:
```env
SUPABASE_URL=...
SUPABASE_KEY=...
GROQ_API_KEY=...
```

## Supabase Setup
1. Create a Supabase project.
2. Open the SQL editor.
3. Run `sql/schema.sql`.
4. Confirm both `documents` and `document_chunks` tables exist.
5. Confirm the API key in `.env` can insert, select, and delete rows.

## How To Run
1. Start the FastAPI backend from the project root.
```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
2. In a second terminal, start Streamlit.
```powershell
streamlit run frontend/app.py
```
3. Open the Streamlit URL shown in the terminal, usually `http://localhost:8501`.
4. Upload one or more hospital PDFs.
5. Click `Process Document(s)`.
6. Ask questions in the chat box.

## Demo Flow
Use `C:/Users/Nimish/Downloads/Ai_ml_assignment.pdf` as the primary hospital document.

Suggested checks:
- `What are OPD timings?`
- `Who is the cardiologist?`
- `What is MRI cost?`
- `ICU cost per day?`
- `Emergency number?`
- `Can I cancel appointment within 24 hours?`

Expected behavior:
- Answers only use uploaded document evidence
- Citations show `filename - page`
- Source chunk panels show the exact retrieved text
- Missing facts return exactly:
```text
I don't have that information in the provided document.
```

## API Summary
### `POST /upload`
- Upload a single PDF and index it into Supabase.

### `GET /documents`
- List all uploaded documents and indexing metadata.

### `DELETE /documents/{document_id}`
- Remove a document and its chunks.

### `POST /query`
Request body:
```json
{
  "question": "What are OPD timings?",
  "document_ids": [],
  "chat_history": [
    {"role": "user", "content": "What are OPD timings?"}
  ]
}
```

### `GET /health`
- Returns configuration and embedding readiness.

## Screenshots
Add screenshots after running the app:
- Upload/indexing view
- Multi-document chat view
- Answer with citations and source text expanded

## Troubleshooting
- Missing `.env` values:
  The backend will fail fast if `SUPABASE_URL`, `SUPABASE_KEY`, or `GROQ_API_KEY` are missing.
- Supabase SQL not applied:
  Retrieval and inserts will fail until `sql/schema.sql` has been run.
- First embedding load is slow:
  `BAAI/bge-small-en-v1.5` downloads on first use and is cached afterward.
- Empty or scanned PDF:
  If text extraction fails, the upload endpoint returns a clear error.
- Query returns the fallback:
  That means the answer was not found in the retrieved document context.

## Testing
Run unit tests:
```powershell
pytest
```

## Notes
- Multi-document search is the default behavior. Leave the sidebar filter empty to search across every indexed PDF.
- If different documents contain conflicting facts, the assistant is instructed to keep the answer source-qualified instead of silently combining them.
