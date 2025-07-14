from sentence_transformers import SentenceTransformer
import faiss
import os
import pandas as pd

class DocumentRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def build_faiss_index(self, embeddings, texts):
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, texts

    def load_index(self, path='docs/faiss_index/index.faiss'):
        index = faiss.read_index(path)
        return index
