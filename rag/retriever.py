from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    """A retriever wrapper that uses sentence-transformers + faiss if available.
    Falls back to a TF-IDF cosine similarity vector search otherwise.
    """
    def __init__(self, embed_model_name: str = "BAAI/bge-small-en", device: str = "cpu"):
        self.embed_model_name = embed_model_name
        self.device = device
        self.embed_model = None
        self.index = None
        self.kb_docs: List[Dict[str, Any]] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        if SentenceTransformer is not None:
            try:
                # lazy load: will be loaded when build_index is called
                self.embed_model = None
            except Exception:
                self.embed_model = None

    def build_index(self, kb_docs: List[Dict[str, Any]]):
        self.kb_docs = kb_docs
        texts = [d["text"] for d in kb_docs]

        if SentenceTransformer is not None:
            # Try to load model lazily
            try:
                if self.embed_model is None:
                    # allow a device hint to SentenceTransformer
                    self.embed_model = SentenceTransformer(
                        self.embed_model_name, device=self.device
                    )
                embeddings = self.embed_model.encode(
                    texts, convert_to_numpy=True, normalize_embeddings=True
                ).astype("float32")
            except Exception:
                embeddings = None
            if embeddings is not None and faiss is not None:
                dim = embeddings.shape[1]
                index = faiss.IndexHNSWFlat(
                    dim, 32, faiss.METRIC_INNER_PRODUCT
                )
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 64
                index.add(embeddings)
                self.index = index
                self.embeddings = embeddings
                return

        # Fallback: TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.index = None

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is not None and hasattr(self, "embeddings"):
            if SentenceTransformer is None:
                raise RuntimeError(
                    "SentenceTransformer is required for faiss mode"
                )
            q_emb = self.embed_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            D, idxs_arr = self.index.search(
                q_emb, min(k, self.index.ntotal)
            )
            idxs = idxs_arr[0]
            scores = D[0]
            results = []
            for rank, (i, score) in enumerate(zip(idxs, scores)):
                doc = self.kb_docs[i]
                results.append({
                    "rank": rank,
                    "score": float(score),
                    "id": doc["id"],
                    "text": doc["text"],
                })
            return results

        # TF-IDF fallback
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError(
                "Index not built. Call build_index(kb_docs) first."
            )
        q_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        idxs = np.argsort(sims)[::-1][:k]
        results = []
        for rank, i in enumerate(idxs):
            results.append({
                "rank": rank,
                "score": float(sims[i]),
                "id": self.kb_docs[i]["id"],
                "text": self.kb_docs[i]["text"],
            })
        return results
