import os
import sys
import uuid
import logging
from collections import OrderedDict

from dotenv import load_dotenv
from flask import Flask, request, jsonify, session

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

# In-memory session store: session_id -> {"store": VectorStore, "df": DataFrame}
# Fine for a demo/resume project. Swap for Redis or on-disk persistence
# if this needs to survive a server restart or run with multiple workers.
# Bounded + FIFO eviction so a long-running server doesn't leak memory
# indefinitely as people upload files -- real fix for real deployment is
# a TTL-based store (Redis with expiry), this is a simple stopgap.
MAX_SESSIONS = 100
SESSIONS = OrderedDict()


def _store_session(session_id: str, data: dict):
    SESSIONS[session_id] = data
    if len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)  # evict oldest


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/upload", methods=["POST"])
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
    app.run(debug=True)