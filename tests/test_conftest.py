"""Stub heavy/networked deps so tests run fast without real API keys or
downloading the embedding model -- these tests only exercise the pure
pandas computation logic in core/agent.py, not actual LLM or embedding calls."""
import sys
import types

if "groq" not in sys.modules:
    groq_mod = types.ModuleType("groq")
    groq_mod.Groq = lambda **kwargs: None
    sys.modules["groq"] = groq_mod

if "fastembed" not in sys.modules:
    fastembed_mod = types.ModuleType("fastembed")
    fastembed_mod.TextEmbedding = object
    sys.modules["fastembed"] = fastembed_mod