import os
import glob
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import argparse
import os

def read_texts(root: str) -> List[Tuple[str, str]]:
    exts = ("*.txt", "*.md", "*.html")
    files = []
    for p in exts:
        files.extend(glob.glob(os.path.join(root, p)))
    docs = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((fp, f.read()))
        except Exception:
            continue
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        if end == n:
            break
        i = max(end - overlap, 0)
    return chunks


class ChromaStore:
    def __init__(self, persist_dir: str, collection: str = "docs"):
        
        self.client = chromadb.Client(Settings(is_persistent=True, persist_directory=persist_dir))
        self.col = self.client.get_or_create_collection(collection_name=collection, metadata={"hnsw:space": "cosine"})

    def add(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict]):
        self.col.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(self, query_embed: List[float], k: int = 5) -> List[dict]:
        res = self.col.query(query_embeddings=[query_embed], n_results=k)
        out = []
        for i in range(len(res["ids"][0])):
            out.append({"id": res["ids"][0][i], "metadata": res["metadatas"][0][i]})
        return out


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int):
        self.qm = qm
        self.client = QdrantClient(url=url)
        self.collection = collection
        if collection not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(collection_name=collection, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))

    def add(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict]):

        points = [qm.PointStruct(id=ids[i], vector=embeddings[i], payload=metadatas[i]) for i in range(len(ids))]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_embed: List[float], k: int = 5) -> List[dict]:
        
        res = self.client.search(collection_name=self.collection, query_vector=query_embed, limit=k)
        out = []
        for p in res:
            out.append({"id": str(p.id), "metadata": p.payload})
        return out


def build_index(docs_dir: str, backend: str = "chroma", persist: str = ".chroma", collection: str = "docs"):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    docs = read_texts(docs_dir)
    ids, embs, metas = [], [], []
    for i, (path, text) in enumerate(docs):
        chunks = chunk_text(text)
        vecs = model.encode(chunks, normalize_embeddings=True).tolist()
        for j, v in enumerate(vecs):
            ids.append(f"{i}-{j}")
            embs.append(v)
            metas.append({"path": path, "chunk": chunks[j]})
    if backend == "qdrant":
        store = QdrantStore(os.environ.get("QDRANT_URL", "http://localhost:6333"), collection, len(embs[0]) if embs else 384)
    else:
        store = ChromaStore(persist_dir=persist, collection=collection)
    if ids:
        store.add(ids, embs, metas)
    return store, model


def get_store(backend: str = "chroma", persist: str = ".chroma", collection: str = "docs", dim: int = 384):
    if backend == "qdrant":
        return QdrantStore(os.environ.get("QDRANT_URL", "http://localhost:6333"), collection, dim)
    return ChromaStore(persist_dir=persist, collection=collection)


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="llm-finetune/data/manuals")
    ap.add_argument("--backend", default="chroma", choices=["chroma", "qdrant"])
    ap.add_argument("--persist", default=".chroma")
    ap.add_argument("--collection", default="docs")
    args = ap.parse_args()
    build_index(args.docs, backend=args.backend, persist=args.persist, collection=args.collection)
