from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def load_model():
    """Load the embedding model once per process and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_embeddings(chunks: list[str]) -> np.ndarray:
    model = load_model()
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings, dtype='float32')