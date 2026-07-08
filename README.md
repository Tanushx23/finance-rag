# Finance RAG

Upload a CSV of transactions and ask questions about your spending in natural language — powered by a small tool-calling agent, not just plain RAG.

**Live demo (Streamlit version):** [link](https://tanushx23-finance-rag-app-iferdt.streamlit.app/)

## Why an agent, not just RAG

Plain RAG asks the LLM to add up retrieved numbers itself — unreliable. Testing "how much did I spend on food?" once produced:

> "You spent 1750 on Food... 400+600+350+500=1850... the correct total is 1850."

Fix: a tool-calling agent that routes each question to either:
- **`compute_total_spent` / `get_top_transaction`** — exact pandas computation for sums, counts, date ranges, max/min
- **`semantic_search`** — FAISS retrieval for genuinely open-ended questions

The LLM only picks a tool and phrases the result — it never does the math.

## Stack
Flask · pandas · sentence-transformers (`all-MiniLM-L6-v2`) · FAISS · Groq (`openai/gpt-oss-120b`)

## Setup
```bash
python -m venv venv && source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY (free at console.groq.com/keys)
python app.py
```

## API
- `POST /upload` — multipart `file` (CSV with `date, amount, category, description`) → `{session_id, transactions_loaded}`
- `POST /query` — `{"session_id", "question"}` → `{"answer", "tool_used"}`
- `GET /health` → `{"status": "ok"}`

## Testing
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```
17 tests, each mapped to a real bug found while building this (arithmetic errors, category mismatches, date-year assumptions, undercounting).

## Key design decisions
- **In-memory sessions** (bounded, FIFO eviction) — fine for a demo; a real deployment would use Redis with TTL.
- **Category matching**: LLM maps user wording onto the real category list (JSON schema enum), with a `difflib` fuzzy fallback for typos/casing. Unrecognized categories return an explicit error instead of a silent ₹0.
- **Date disambiguation**: the LLM is told the data's actual date range, so "Jan 1–15" with no year doesn't default to the current year.
- **Model**: switched from a deprecated Groq model to `openai/gpt-oss-120b`, with a one-retry safety net for occasional malformed tool calls.

## Next steps
- Multi-turn conversation memory
- Dedicated category-breakdown tool
- Redis-backed sessions for persistence across restarts