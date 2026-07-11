# Finance RAG

Upload a CSV of transactions and ask questions about your spending in natural language — powered by a small tool-calling agent, not just plain RAG.

**Live app:** https://finance-rag-oj4n.onrender.com
*(free-tier hosting — may take 20-30s to wake up if idle)*

## Why an agent, not just RAG

Plain RAG asks the LLM to add up retrieved numbers itself — unreliable. Testing "how much did I spend on food?" once produced:

> "You spent 1750 on Food... 400+600+350+500=1850... the correct total is 1850."

Fix: a tool-calling agent that routes each question to one of four tools:
- **`compute_total_spent`** — exact pandas computation for totals, counts, averages, and date ranges
- **`get_top_transaction`** — deterministic max/min lookup for "most expensive" / "cheapest" questions
- **`get_category_breakdown`** — ranked spending by category, with exact percentages
- **`semantic_search`** — FAISS retrieval for genuinely open-ended questions

The LLM only picks a tool and phrases the result — it never does the math.

## Stack
Flask · pandas · fastembed (ONNX embeddings) · FAISS · Groq (`openai/gpt-oss-120b`) · vanilla HTML/CSS/JS frontend

## Setup
```bash
python -m venv venv && source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY (free at console.groq.com/keys)
python app.py
```
Open `http://127.0.0.1:5000` — upload a CSV, then ask questions.

## API
- `GET /` — web UI
- `POST /upload` — multipart `file` (CSV with `date, amount, category, description`) → `{session_id, transactions_loaded}` *(rate limited: 10/hour per IP)*
- `POST /query` — `{"question"}` (session tracked via cookie) → `{"answer", "tool_used"}` *(rate limited: 20/hour per IP)*
- `GET /sample-csv` — downloads example transaction data for visitors without their own
- `GET /health` → `{"status": "ok"}`

## Testing
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```
21 tests, each mapped to a real bug found while building this (arithmetic errors, category mismatches, date-year assumptions, undercounting, unreliable averaging, incorrect category rankings).

## Deployment
Dockerized (see `Dockerfile`) and deployed on Render's free tier. Two real issues surfaced only at deployment and are fixed in code:
- **Memory limit**: the original `sentence-transformers`/PyTorch embedding stack exceeded Render's 512MB free-tier RAM cap and got OOM-killed mid-request. Switched to `fastembed` (same model, ONNX runtime) — same embedding quality, a fraction of the memory.
- **Single worker required**: `SESSIONS` is an in-memory dict, so the Docker image runs gunicorn with `--workers 1` — multiple workers would each have their own copy and randomly "lose" sessions.

## Key design decisions
- **In-memory sessions** (bounded, FIFO eviction) — fine for a demo; a real deployment would use Redis with TTL. On Render's free tier, an idle container sleeping after 15 min also wipes sessions.
- **Category matching**: LLM maps user wording onto the real category list (JSON schema enum), with a `difflib` fuzzy fallback for typos/casing. Unrecognized categories return an explicit error instead of a silent ₹0.
- **Date disambiguation**: the LLM is told the data's actual date range, so "Jan 1–15" with no year doesn't default to the current year.
- **Averages computed in pandas**, not by the model — same reasoning as totals; division that looks clean in one test case isn't guaranteed to always be correct.
- **Model**: switched from a deprecated Groq model to `openai/gpt-oss-120b`, with a one-retry safety net for occasional malformed tool calls.
- **Rate limited** (Flask-Limiter, in-memory): 10 uploads/hour and 20 questions/hour per IP, since this is a public deployment using a personal API key.

## Next steps
- Multi-turn conversation memory
- Redis-backed sessions for persistence across restarts/sleep cycles