# Fully-Free RAG Pipeline — Local Notebook + Script

This repository contains a ready-to-run Retrieval-Augmented Generation (RAG) pipeline blueprint and a small demo runner. It includes a Jupyter notebook, helper modules, a CLI runner script, and tests. Everything is designed to be runnable locally and free — no OpenAI or paid APIs.

Overview
- A small, local-first RAG pipeline that demonstrates:
  - KB ingestion and chunking (text chunking utilities).
  - Dense embeddings via SentenceTransformers (BGE-small-en recommended when available).
  - FAISS-based ANN indexing (HNSW).
  - A small Retriever interface that returns top-k hits.
  - Optional local LLM generation (if resources and VRAM permit).
- The pipeline falls back to a TF-IDF-style retrieval if a real embedding stack (sentence-transformers and FAISS) is not available.

Quick start
1. (Optional) Create and activate a Python virtual environment.
2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```
3. Open `free_rag_pipeline.ipynb` and run cells top-to-bottom.

Or run the demo CLI (toy demo that does not require GPU or large models):
```bash
python run_rag.py
```

CLI usage notes
- Use `--force-device cpu` or `--force-device cuda` to force device selection.
  - `--force-device cuda` will try to run on GPU but will fall back to CPU if your venv lacks CUDA-enabled PyTorch.
- Use `--with-llm` to attempt to load a local LLM for generation; omit if you have limited VRAM.

Files included
- `free_rag_pipeline.ipynb`: Notebook blueprint with cell-by-cell content.
- `run_rag.py`: Minimal CLI to run a toy demo using small or mocked embeddings.
- `rag/`: Python package with helper functions used by the notebook/CLI.
  - `rag/utils.py`: chunking helpers (simple_chunk_text, iter_simple_chunk_text).
  - `rag/retriever.py`: retriever implementation (FAISS/embedding fallback to TF-IDF).
  - `rag/device.py`: device diagnostics and helper to choose CUDA vs CPU.
- `tests/`: Unit tests for chunking and retrieval utilities.
- `requirements.txt`: Required packages.

Implementation notes
- Embedding model: the notebook shows an example using `BAAI/bge-small-en` via SentenceTransformers for compact English embeddings; change to your preferred model if needed.
- FAISS index: the example uses an HNSW inner-product index with the following parameters:
  - Index: `IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)`
  - Construction/search tuning: `efConstruction=200`, `efSearch=64`
- Retriever: `rag.retriever.Retriever` builds an index from KB documents and executes top-k retrieval. If `sentence-transformers` or `faiss` are unavailable, the retriever will fall back to a basic numpy/TF-IDF retrieval mode.
- Chunking: use `rag.utils.simple_chunk_text` to chunk text into smaller passages; `iter_simple_chunk_text` yields streaming chunks for large datasets.

Example run and results
Below is a typical notebook/CLI run with toy documents (works on CPU or CUDA if available). These example outputs were produced in a sample run of the notebook:

1) Device diagnostics (example on a machine with CUDA):
```
Using device: cuda
{'torch_version': '2.7.1+cu118', 'cuda_version': '11.8', 'cuda_available': True, 'device_name': 'NVIDIA GeForce RTX 4060', 'total_memory': 8585216000, 'major': 8, 'minor': 9}
```

2) KB preparation and chunking:
```
Loaded 3 raw docs. Chunked to 6 docs.
```

3) Embedding/FAISS index creation:
```
FAISS index created, size: 6
```

4) Example retrieval for query "What is RAG?":
```
0 0.859682559967041 doc_001_chunk_0 -> Retrieval-Augmented Generation (RAG) combines a retriever with a generator to an...
1 0.8120253086090088 doc_002_chunk_1 -> The BGE-small-en model is a compact, general-purpose English embedding model useful for...
2 0.7771009206771851 doc_003_chunk_1 -> FAISS is a library for efficient similarity search and clustering of dense vectors...
```

CLI run with `--with-llm`
The `run_rag.py` script can also load a local LLM and generate an answer when invoked with `--with-llm` (if your environment has sufficient resources). Example output from running the CLI with a CUDA-enabled environment:

```
Query: What is RAG?
Top docs:
[
  {
    "rank": 0,
    "score": 0.859682559967041,
    "id": "doc_001_chunk_0",
    "text": "Retrieval-Augmented Generation (RAG) combines a retriever with a generator to answer questions based on external documents."
  },
  {
    "rank": 1,
    "score": 0.8120253086090088,
    "id": "doc_002_chunk_1",
    "text": "el useful for semantic search and retrieval tasks."
  },
  {
    "rank": 2,
    "score": 0.7771009206771851,
    "id": "doc_003_chunk_1",
    "text": "f dense vectors, widely used for vector databases."
  }
]
Attempting to generate an answer using LocalLLM
`torch_dtype` is deprecated! Use `dtype` instead!
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


=== ANSWER ===
Context:
Retrieval-Augmented Generation (RAG) combines a retriever with a generator to answer questions based on external documents.

el useful for semantic search and retrieval tasks.

f dense vectors, widely used for vector databases.

Question: What is RAG?
Answer: RAG is a semantic search engine that uses a combination of a query language and a generator to generate a query.
```

Notes & recommendations
- For production workloads with large KBs:
  - Prefer streaming chunking (iter_simple_chunk_text) to avoid large memory usage.
  - Persist embeddings and FAISS indexes to disk and use sharding or a remote vector DB if needed.
- For LLM generation:
  - If using HF Transformers, `device_map="auto"` with `accelerate` installed helps place the model on GPU.
  - Consider quantization (bitsandbytes) and/or smaller models to reduce VRAM usage.
- If FAISS GPU is not installed, you can still index with `faiss-cpu` or use the TF-IDF fallback.

Troubleshooting
- Confirm GPU visibility with:
```cmd
nvidia-smi
```
- Confirm CUDA-enabled PyTorch is installed:
```py
import torch
print(torch.cuda.is_available(), torch.__version__, torch.version.cuda)
```

License
This project is provided as-is (no license specified).