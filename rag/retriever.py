"""RAG retriever: embeds the medical corpus and returns top-k passages for a query.

TODO:
- Point corpus_path (see configs/*.yaml) at an actual medical knowledge source
  (e.g. MedQA textbooks corpus, PubMed abstracts, or a curated subset).
- Build the FAISS index once with build_index(), then reuse it across runs.
"""

from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class Retriever:
    def __init__(self, corpus_path: str, embedding_model: str = "BAAI/bge-small-en", top_k: int = 5):
        self.corpus_path = Path(corpus_path)
        self.top_k = top_k
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.passages: List[str] = []

    def build_index(self) -> None:
        """Load all text files under corpus_path, embed them, and build a FAISS index."""
        self.passages = self._load_corpus()
        embeddings = self.model.encode(self.passages, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype="float32"))

    def _load_corpus(self) -> List[str]:
        texts = []
        for file in self.corpus_path.glob("**/*.txt"):
            texts.append(file.read_text(encoding="utf-8"))
        return texts

    def retrieve(self, query: str) -> List[str]:
        if self.index is None:
            raise RuntimeError("Call build_index() before retrieve().")
        query_emb = self.model.encode([query], normalize_embeddings=True)
        _, indices = self.index.search(np.array(query_emb, dtype="float32"), self.top_k)
        return [self.passages[i] for i in indices[0] if i < len(self.passages)]

