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

    def search_by_threshold(
        self,
        query_embedding: np.ndarray,
        distance_threshold: float = 1.2,
        max_results: int = 50,
        min_results: int = 3,
    ) -> list[str]:
        query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)

        lims, distances, indices = self.index.range_search(
            query_embedding, thresh=distance_threshold
        )
        matched_indices = indices[lims[0]:lims[1]]
        matched_distances = distances[lims[0]:lims[1]]

        order = np.argsort(matched_distances)
        matched_indices = matched_indices[order]

        if len(matched_indices) < min_results:
            _, fallback_indices = self.index.search(query_embedding, min_results)
            matched_indices = fallback_indices[0]

        matched_indices = matched_indices[:max_results]
        results = [self.chunks[i] for i in matched_indices if i < len(self.chunks)]
        return results