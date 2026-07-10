import os
import sys
import uuid
import logging
from collections import OrderedDict

from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, render_template, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from core.parser import load_transactions_df, build_chunks
from core.embeddings import get_embeddings
from core.vector_store import VectorStore
from core.agent import answer_question

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance_rag")

# Fail loudly at startup if the Groq key is missing, instead of letting the
# app boot fine and then fail confusingly on the first real /query request.
if not os.getenv("GROQ_API_KEY"):
    logger.error(
        "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
        "from https://console.groq.com/keys before running the app."
    )
    sys.exit(1)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# Rate limiting: this app is public, uses a personal GROQ_API_KEY, and
# /upload builds a fresh embedding index each time (real compute cost).
# Without a limit, anyone (or a bot) hitting these endpoints repeatedly
# could burn through the API quota or fill memory with sessions. In-memory
# storage is fine here since we're already single-worker (see Dockerfile)
# and sessions are already in-memory -- consistent with the rest of the
# app's demo-scale tradeoffs.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per hour"],
    storage_uri="memory://",
)

# Warm up the embedding model at startup, not on the first real request.
# Without this, the model's first download/load happens mid-request --
# on Windows specifically, this caused a real bug: Flask's debug
# auto-reloader saw the newly-downloaded model files appear on disk mid-
# download and restarted the whole server, killing the in-flight request
# (showed up in the browser as a generic "network error"). Loading it here
# means the download/load is finished before the server ever accepts a

# request.
logger.info("Warming up embedding model...")
get_embeddings(["warmup"])
logger.info("Embedding model ready.")
MAX_SESSIONS = 100
SESSIONS = OrderedDict()


def _store_session(session_id: str, data: dict):
    SESSIONS[session_id] = data
    if len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)  # evict oldest

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sample-csv")
def sample_csv():
    # Lets visitors without their own transaction data try the live demo
    # end-to-end -- reuses the same file the test suite is built against.
    return send_from_directory(
        app.root_path, "test_transactions.csv", as_attachment=True
    )


@app.route("/health")
@limiter.exempt
def health():
    return jsonify({"status": "ok"})


@app.route("/upload", methods=["POST"])
@limiter.limit("10 per hour")
def upload():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a valid CSV file"}), 400

    try:
        df = load_transactions_df(file)
        chunks = build_chunks(df)
    except Exception as e:
        logger.exception("Failed to parse uploaded CSV")
        return jsonify({
            "error": "Failed to parse CSV. Make sure it has date, amount, "
                    "category, and description columns.",
            "detail": str(e),
        }), 400

    if len(chunks) == 0:
        return jsonify({"error": "No transactions found in CSV. Please check your file format."}), 400

    try:
        embeddings = get_embeddings(chunks)
        store = VectorStore()
        store.build(chunks, embeddings)
    except Exception as e:
        logger.exception("Failed to build vector store")
        return jsonify({"error": "Failed to process transactions", "detail": str(e)}), 500

    session_id = str(uuid.uuid4())
    _store_session(session_id, {"store": store, "df": df})
    session["session_id"] = session_id

    return jsonify({"session_id": session_id, "transactions_loaded": len(chunks)})


@app.route("/query", methods=["POST"])
@limiter.limit("20 per hour")
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    session_id = session.get("session_id") or data.get("session_id")

    if not question:
        return jsonify({"error": "question is required"}), 400
    if not session_id or session_id not in SESSIONS:
        return jsonify({"error": "No active session — upload a CSV first"}), 400

    session_data = SESSIONS[session_id]

    try:
        result = answer_question(question, session_data["df"], session_data["store"])
    except Exception as e:
        logger.exception("Failed to answer query")
        return jsonify({"error": "Failed to generate answer", "detail": str(e)}), 500

    return jsonify({
        "answer": result["answer"],
        "tool_used": result["tool_used"],
    })


if __name__ == "__main__":
    # use_reloader=False: the auto-reloader restarts the whole process
    # whenever it sees files change on disk -- which includes files written
    # by fastembed/faiss during normal operation, not just your own code
    # edits. That caused a real bug (see the warmup comment above). Debug
    # error pages still work fine with the reloader off; you'll just need
    # to manually restart (Ctrl+C, rerun) after editing code.
    app.run(debug=True, use_reloader=False)