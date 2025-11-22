# CONTEXT.md — Fully-Free RAG Pipeline (Jupyter Notebook Blueprint)

This document is your **blueprint** for building a **0-cost RAG system** in a **single Jupyter notebook**, running **fully locally** on your machine (e.g., RTX 4060).

Key design choices:

- **Embeddings**: `BAAI/bge-small-en` (free, sentence-transformers)
- **Vector store**: `faiss-cpu` (free)
- **LLM (generator)**: a small, free, local instruct model via `transformers`, e.g.:
  - `Qwen/Qwen2.5-1.5B-Instruct` (lighter, easier for 4060)
  - You can later swap to any other local instruct model.

No OpenAI, no paid API calls, no cloud usage required.

The notebook will be structured as:

1. Environment & installs
2. Imports & device setup
3. Data loading & chunking
4. Embedding model & FAISS index
5. Retriever function
6. Local LLM (generator) setup
7. RAG answer function
8. Test queries
9. (Optional) Evaluation helpers

You can literally copy/paste each code block into separate notebook cells.

---

## 1. Notebook Cell 1 — Environment Setup (optional in local venv)

If you are in a fresh environment, install dependencies:

```python
# CELL 1: Install dependencies (run once)
%pip install --upgrade pip

%pip install \
    sentence-transformers \
    faiss-cpu \
    torch \
    transformers \
    accelerate \
    einops \
    tqdm
````

> If you’re offline, you’ll need to have these wheels downloaded somewhere or pre-installed. For normal usage, you run this once with internet to download models and libs, then reuse the environment offline later.

---

## 2. Notebook Cell 2 — Imports & Device Setup

```python
# CELL 2: Imports & device setup

import os
from typing import List, Dict, Any

import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

# Detect device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
```

---

## 3. Notebook Cell 3 — Data Preparation (Your Knowledge Base)

This is where you define your **knowledge base documents**.

For now, we’ll mock it with a few example texts. Replace this with your own data loading from files / DB.

```python
# CELL 3: Define / load knowledge base documents

# Example toy documents — replace with your own
raw_docs = [
    {
        "id": "doc_001",
        "text": "Retrieval-Augmented Generation (RAG) combines a retriever with a generator to answer questions based on external documents."
    },
    {
        "id": "doc_002",
        "text": "The BGE-small-en model is a compact, general-purpose English embedding model useful for semantic search and retrieval tasks."
    },
    {
        "id": "doc_003",
        "text": "FAISS is a library for efficient similarity search and clustering of dense vectors, widely used for vector databases."
    },
]

print(f"Loaded {len(raw_docs)} raw documents.")
```

### 3.1 Chunking (if your docs are long)

If your real docs are long (articles, PDFs), chunk them. You can start with simple rule-based splitting.

```python
# CELL 3b: Simple chunking utility (optional, if you need it)

def simple_chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Suitable as a first pass; you can replace with token-based later.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if start >= n:
            break
    return [c for c in chunks if c]

kb_docs = []
for d in raw_docs:
    chunks = simple_chunk_text(d["text"], max_chars=400, overlap=50)
    for idx, ch in enumerate(chunks):
        kb_docs.append({
            "id": f"{d['id']}_chunk_{idx}",
            "text": ch
        })

print(f"After chunking, we have {len(kb_docs)} KB documents.")
```

---

## 4. Notebook Cell 4 — Embedding Model & Vectorization

Use `BAAI/bge-small-en` for embeddings (free).

```python
# CELL 4: Load embedding model and compute embeddings

EMB_MODEL_NAME = "BAAI/bge-small-en"

embed_model = SentenceTransformer(EMB_MODEL_NAME, device=device)
print("Loaded embedding model:", EMB_MODEL_NAME)

texts = [d["text"] for d in kb_docs]

embeddings = embed_model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # cosine similarity via inner product
)

embeddings = embeddings.astype("float32")  # faiss expects float32
embeddings.shape
```

---

## 5. Notebook Cell 5 — FAISS Index Construction (Free Vector DB)

Build a FAISS HNSW index for ANN search.

```python
# CELL 5: Build FAISS HNSW index

dim = embeddings.shape[1]  # 384 for bge-small-en
m = 32  # HNSW graph parameter

index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64

index.add(embeddings)
print("FAISS index size:", index.ntotal)
```

You can optionally save it:

```python
# Optional persistence
faiss.write_index(index, "faiss_bge_small_hnsw.index")
np.save("kb_embeddings.npy", embeddings)
```

---

## 6. Notebook Cell 6 — Retriever Function

```python
# CELL 6: Retriever function

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve top-k similar KB chunks.
    """
    q_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    
    D, I = index.search(q_emb, k)  # distances, indices
    idxs = I[0]
    scores = D[0]
    
    results = []
    for rank, (i, score) in enumerate(zip(idxs, scores)):
        doc = kb_docs[i]
        results.append({
            "rank": rank,
            "score": float(score),
            "id": doc["id"],
            "text": doc["text"]
        })
    return results

# Quick smoke test
test_retrieval = retrieve("What is RAG?", k=3)
for r in test_retrieval:
    print(r["rank"], r["score"], r["id"], "->", r["text"][:80], "...")
```

---

## 7. Notebook Cell 7 — Local LLM (Free Generator)

Here, we pick a **small instruct model** that can run on your RTX 4060. Example:

* `Qwen/Qwen2.5-1.5B-Instruct`

You can swap this for any other HF model.

```python
# CELL 7: Load local LLM (free instruct model)

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()
print("Loaded LLM:", LLM_MODEL_NAME, "on", device)
```

> If you run out of VRAM, you can:
>
> * Use a smaller model, or
> * Load in 8-bit / 4-bit quantization (needs `bitsandbytes` + proper flags).

---

## 8. Notebook Cell 8 — LLM Generation Helper

```python
# CELL 8: Generation helper

def generate_llm_response(prompt: str, max_new_tokens: int = 256, temperature: float = 0.3) -> str:
    """
    Basic text generation wrapper for the local LLM.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the original prompt if the model echoes
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()
    return decoded.strip()
```

---

## 9. Notebook Cell 9 — RAG Prompt Construction

We now build the RAG “chain”:

1. Retrieve top-k docs for query.
2. Build a prompt with **context + question**.
3. Call local LLM to answer based on context.

```python
# CELL 9: RAG prompt builder

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context.\n"
    "If the answer is not contained in the context, say: \"I’m not sure based on the given documents.\"\n"
    "Do NOT hallucinate.\n\n"
)

def build_rag_prompt(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for r in retrieved_docs:
        context_blocks.append(f"[Document {r['rank']+1}]\n{r['text']}")
    context_str = "\n\n".join(context_blocks)
    
    prompt = (
        SYSTEM_PROMPT +
        "Context:\n" + context_str + "\n\n" +
        f"Question:\n{query}\n\n" +
        "Answer:"
    )
    return prompt
```

---

## 10. Notebook Cell 10 — RAG Answer Function

```python
# CELL 10: Full RAG pipeline

def rag_answer(query: str, k: int = 5, max_new_tokens: int = 256) -> Dict[str, Any]:
    retrieved = retrieve(query, k=k)
    prompt = build_rag_prompt(query, retrieved)
    answer = generate_llm_response(prompt, max_new_tokens=max_new_tokens)
    return {
        "query": query,
        "answer": answer,
        "retrieved": retrieved,
        "prompt": prompt
    }

# Quick test
result = rag_answer("What is Retrieval-Augmented Generation?", k=3)
print("=== Answer ===")
print(result["answer"])
print("\n=== Retrieved docs ===")
for r in result["retrieved"]:
    print(f"[{r['rank']}] score={r['score']:.3f} id={r['id']}")
    print(r["text"])
    print("----")
```

---

## 11. Notebook Cell 11 — Multiple Queries / Interactive Loop (Optional)

```python
# CELL 11: Interactive CLI loop inside notebook (optional)

while True:
    q = input("\nAsk a question (or type 'quit'): ").strip()
    if q.lower() in ("quit", "exit"):
        break
    out = rag_answer(q, k=5, max_new_tokens=200)
    print("\n=== Answer ===")
    print(out["answer"])
    print("\n=== Top documents used ===")
    for r in out["retrieved"]:
        print(f"- [rank {r['rank']}] score={r['score']:.3f} id={r['id']}")
```

---

## 12. Summary — What You’ve Built (Fully Free RAG)

In this notebook, you:

1. **Prepared documents** (`raw_docs` → chunked `kb_docs`).
2. **Encoded docs with BGE-small-en** into dense vectors (local, free).
3. **Built a FAISS HNSW index** over embeddings (local, free).
4. **Implemented a retriever** (`retrieve(query, k)`).
5. **Loaded a local instruct model** (`Qwen2.5-1.5B-Instruct`) as your generator (local, free).
6. **Constructed a RAG prompt** combining **context + query** under a clear system instruction.
7. **Created `rag_answer`**, which:

   * Retrieves top-k docs,
   * Builds a context-aware prompt,
   * Calls the local LLM to generate an answer.
8. (Optionally) Added an **interactive loop** to ask questions against your KB.

Everything runs **fully on your machine**: **no OpenAI, no Colab, no per-token cost**.

---

## 13. How to Turn This Into a `.ipynb` File

1. Open **Jupyter Notebook** or **VS Code + Jupyter**.
2. Create a new notebook: `free_rag_pipeline.ipynb`.
3. For each numbered section above:

   * Create a new cell.
   * Paste the corresponding code block.
4. Run cells top-to-bottom.

If you want, you can also:

* Add markdown cells before each code block, copying the headings/explanations here.
* Save the notebook; that’s your fully-free RAG implementation.

---s