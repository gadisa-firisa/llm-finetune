import os
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import sys
sys.path.append(os.path.dirname(__file__))
from .model import ChatModel
from .data import get_store
from sentence_transformers import SentenceTransformer
import uvicorn
from fastapi.templating import Jinja2Templates

def format_rag_prompt(query: str, contexts: List[str]):
    system = "You answer questions using the provided context. If the answer is not in the context, say you don't know."
    ctx = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)])
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"},
    ]

    return messages


app = FastAPI()
chat_model = None
embedder = None
store = None

templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def startup():
    global chat_model, embedder, store
    base = os.environ.get("BASE_MODEL", "Qwen/Qwen3-3B-Instruct")
    adapter = os.environ.get("ADAPTER", None)
    backend = os.environ.get("VECTOR_BACKEND", "chroma")
    persist = os.environ.get("VECTOR_DIR", ".chroma")
    collection = os.environ.get("VECTOR_COLLECTION", "docs")
    chat_model = ChatModel(base, adapter)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    store = get_store(backend=backend, persist=persist, collection=collection, dim=384)


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"answer": ""})
    qv = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    hits = store.search(qv, k=4)
    contexts = [h["metadata"]["chunk"] for h in hits]
    messages = format_rag_prompt(query, contexts)
    answer = chat_model.generate(messages, max_new_tokens=256)
    return JSONResponse({"answer": answer, "contexts": contexts})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
