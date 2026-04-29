# Logistics Document QA POC

FastAPI-based proof of concept for a Transportation Management System assistant that:

- accepts logistics PDFs
- extracts sections and tables separately
- stores embeddings in local Chroma
- answers only from retrieved evidence
- applies basic guardrails
- returns a confidence score with citations
- falls back to quoted retrieved evidence if OpenAI generation is unavailable
- extracts structured shipment data automatically after upload

<img width="1365" height="975" alt="image" src="https://github.com/user-attachments/assets/8dab1bc4-27ce-4b05-9cdb-e7d36c7e557e" />
<img width="1371" height="971" alt="image" src="https://github.com/user-attachments/assets/b8da8ddc-72f2-416b-bcd6-ea87ab4d8749" />

## Requirements

- Python 3.12+
- `uv`
- `OPENAI_API_KEY`

## Setup

```bash
cp .env.example .env
uv sync
```

Set `OPENAI_API_KEY` in `.env`.

The default embedding model is `text-embedding-3-small`. Override it with `OPENAI_EMBEDDING_MODEL` if needed.

The default chat model is `gpt-4o-mini`. Override it with `OPENAI_CHAT_MODEL` if needed.

## Run

```bash
uv run rag2
```

Open `http://127.0.0.1:8000`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (UI)                          │
│  index.html: Upload → Extract → Chat                      │
└─────────────────┬───────────────────────────────────────────┘
                  │ POST /api/documents
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
│  routes_upload.py → ingestion + extraction                  │
│  routes_chat.py   → retrieval + QA                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ Ingestion    │   │ Retrieval    │
│ Service     │   │ Service      │
│ (pdfplumber)│   │ (Chroma)     │
└──────┬──────┘   └──────┬──────┘
       ▼                  ▼
┌───────────────┐   ┌───────────────┐
│ Chroma DB    │   │ OpenAI LLM  │
│ (embeddings) │   │ (GPT-4o)   │
└───────────────┘   └───────────────┘
```

### Components

| Component | Responsibility |
|-----------|-------------|
| `routes_upload.py` | Accepts PDF uploads, triggers ingestion, auto-extracts shipment data |
| `routes_chat.py` | Handles Q&A requests with retrieval + guardrails |
| `ingest.py` | Parses PDFs, builds parent/child chunks, metadata |
| `retrieval.py` | Chroma vector store, semantic search with relevance scoring |
| `qa.py` | LLM answer generation with guardrails and confidence scoring |
| `guardrails.py` | Preflight (prompt injection) + evidence threshold checks |
| `scoring.py` | Confidence score calculation based on retrieval quality |
| `extraction.py` | LLM-based structured shipment data extraction |

## Chunking Strategy

The system uses a **two-tier parent/child chunking approach**:

### Parent Chunks
- Each PDF page is extracted as a single parent chunk using `pdfplumber`
- Preserves full page context for better semantic matching
- First page gets section title "Document Overview"

### Child Chunks
- Parent chunks are split into smaller overlapping chunks
- Default chunk size: 1400 characters
- Default overlap: 250 characters
- Enables precise retrieval for specific answers

### Text Cleaning
- Removes repeating headers and footers
- Skips known boilerplate patterns (carrier names, phone numbers, dates)
- Falls back to `unstructured` library if pdfplumber extraction fails

## Retrieval Method

1. **Vector Store**: Chroma with OpenAI embeddings (`text-embedding-3-small`)
2. **Search**: Similarity search with score
3. **Filtering**: Document ID scoped (isolates to uploaded document)
4. **Relevance Normalization**: `1 / (1 + distance)` converts distance to relevance [0,1]
5. **Parent Resolution**: If child chunk retrieved, fetches parent chunk for full context

### Retrieval Config
| Setting | Default | Env Variable |
|---------|---------|------------|
| `retrieval_k` | 6 | `RETRIEVAL_K` |
| `chunk_size` | 1400 | `CHUNK_SIZE` |
| `chunk_overlap` | 250 | `CHUNK_OVERLAP` |

## Guardrails Approach

### Preflight Guardrail
Blocks prompt injection attempts before any processing:

| Pattern | Description |
|---------|------------|
| `ignore (all) previous instructions` | Instruction override attempt |
| `reveal (the) system prompt` | Prompt extraction attempt |
| `developer message` | System prompt bypass |
| `override safety` | Safety bypass attempt |

### Evidence Guardrail
Requires minimum retrieval quality before answering:

```python
if best_relevance < 0.35 or hit_count == 0:
    return "insufficient_evidence"
```

### Response Statuses

| Status | Meaning |
|--------|---------|
| `ok` | Answer grounded in retrieved evidence |
| `blocked` | Question failed preflight guardrail |
| `insufficient_evidence` | Retrieved context too weak |

## Confidence Scoring Method

The confidence score is a weighted combination:

```
score = retrieval_strength * 0.75 + citation_bonus + evidence_bonus - penalty

Where:
- retrieval_strength = average of top 3 relevance scores
- citation_bonus = min(citations / 4, 1) * 0.15
- evidence_bonus = min(retrieved_hits / 6, 1) * 0.1
- penalty = 0.25 if guardrail_status == "insufficient_evidence"
```

### Score Interpretation

| Score Range | Reason |
|------------|--------|
| >= 0.80 | Strong retrieval matches and enough citations |
| 0.55 - 0.79 | Answer grounded, but evidence moderate |
| < 0.55 | Weak or limited supporting evidence |

## Shipment Data Extraction

After document upload, the system automatically extracts structured shipment data using LLM:

| Field | Description |
|-------|------------|
| `shipment_id` | Unique shipment/reference ID |
| `shipper` | Sender company or person |
| `consignee` | Receiver company or person |
| `pickup_datetime` | Scheduled/actual pickup date |
| `delivery_datetime` | Scheduled/actual delivery date |
| `equipment_type` | Container size, trailer type |
| `mode` | FTL, LTL, air, ocean, rail |
| `rate` | Shipping cost (numeric) |
| `currency` | USD, CAD, etc. |
| `weight` | Weight in lbs or kgs |
| `carrier_name` | Carrier company name |

The extraction uses the same chat model configured in settings and prompts the LLM to return JSON with these fields.

## API

### Upload a PDF

`POST /api/documents`

Form field:
- `file`: PDF document

Response includes:
- `document_id`: Unique ID for chat reference
- `filename`: Original filename
- `chunk_count`: Number of chunks created
- `section_count`: Unique sections detected
- `shipment_data`: Extracted structured data (or null)

### Ask a Question

`POST /api/chat`

```json
{
  "document_id": "<id from upload>",
  "question": "What are the detention charges after 48 hours?"
}
```

Response includes:
- `answer`: LLM-generated answer
- `confidence`: Score 0-1
- `confidence_reason`: Explanation of score
- `guardrail_status`: ok/blocked/insufficient_evidence
- `citations`: Retrieved chunks with page, relevance
- `contexts`: Full text of retrieved chunks

## Improvement Ideas

1. **Multi-document chat**: Reference multiple documents in single conversation
2. **Table-aware extraction**: Keep table rows as structured data, not plain text
3. **Hardened guardrails**: More injection patterns, rate limiting, PII detection
4. **Streaming responses**: Stream LLM tokens for perceived speed
5. **Document classification**: Auto-detect BOL, Carrier Rate, Invoice, etc.
6. **Export features**: CSV/JSON export for extracted shipment data
7. **Chunk summarization**: Generate summaries for parent chunks
8. **Hybrid search**: Combine vector + keyword (BM25) search
9. **Re-ranking**: Add cross-encoder reranker after initial retrieval
10. **Feedback loop**: Collect user corrections to improve extraction

## Tests

```bash
uv run pytest
```

## Notes

- PDF support is the initial scope.
- Tables are ingested as standalone chunks so rows are not split across chunk boundaries.
- Paragraph chunks stay within their detected section unless a heading change is found.
- If the chat model returns an error, the API still returns the best retrieved evidence with citations instead of failing the request.
- If you switch embedding providers or models, delete `data/chroma` and re-upload documents because vector dimensions may change.
- Guardrails are intentionally lightweight for a POC and should be hardened for production.
