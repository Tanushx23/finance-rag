from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks: list[str]) -> np.ndarray:
    model = load_model()
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings, dtype='float32')