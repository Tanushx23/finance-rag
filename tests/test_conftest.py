"""Stub heavy/networked deps so tests run fast without real API keys or
downloading the embedding model -- these tests only exercise the pure
pandas computation logic in core/agent.py, not actual LLM or embedding calls."""
import sys
import types

if "groq" not in sys.modules:
    groq_mod = types.ModuleType("groq")
    groq_mod.Groq = lambda **kwargs: None
    sys.modules["groq"] = groq_mod

if "sentence_transformers" not in sys.modules:
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = object
    sys.modules["sentence_transformers"] = st_mod