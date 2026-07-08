from fastembed import TextEmbedding
import numpy as np

_model = None

# Same underlying model as before (sentence-transformers/all-MiniLM-L6-v2,
# 384-dim embeddings), but run through fastembed's ONNX runtime instead of
# PyTorch. This was a real fix, not a style preference: on Render's free
# tier (512MB RAM), importing torch + loading the model via
# sentence-transformers pushed memory usage over the limit and got the
# container killed mid-request during /upload. fastembed produces the same
# embedding space with a fraction of the memory footprint.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model():
    """Load the embedding model once per process and reuse it."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def get_embeddings(chunks: list[str]) -> np.ndarray:
    model = load_model()
    embeddings = list(model.embed(chunks))
    return np.array(embeddings, dtype="float32")