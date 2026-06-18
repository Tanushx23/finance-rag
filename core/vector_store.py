import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, chunks: list[str], embeddings: np.ndarray):
        self.chunks = chunks
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[str]:
        query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return results