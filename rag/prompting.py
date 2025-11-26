"""Small runnable CLI demo for the RAG pipeline.

This script uses the `rag` package inside the repo to perform a toy demo.
It intentionally uses small or fallback logic to avoid requiring large model downloads.
"""
from __future__ import annotations
import argparse
import json
from typing import List

from rag.utils import simple_chunk_text
from rag.retriever import Retriever
from rag.device import get_device, cuda_diagnostics
# Import build_prompt if available, otherwise provide a minimal fallback so the script runs.
try:
    from rag.prompting import build_prompt  # type: ignore
except Exception:
    def build_prompt(retrieved: List[dict], query: str) -> str:
        # Minimal fallback prompt builder used when rag.prompting is not present.
        context = "\n\n".join([r.get("text", "") for r in retrieved])
        return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

from rag.generation import LocalLLM


def build_sample_kb() -> List[dict]:
    raw_docs = [
        {"id": "doc_001", "text": "Retrieval-Augmented Generation (RAG) combines a retriever with a generator to answer questions based on external documents."},
        {"id": "doc_002", "text": "The BGE-small-en model is a compact, general-purpose English embedding model useful for semantic search and retrieval tasks."},
        {"id": "doc_003", "text": "FAISS is a library for efficient similarity search and clustering of dense vectors, widely used for vector databases."},
    ]
    kb_docs = []
    for d in raw_docs:
        chunks = simple_chunk_text(d["text"], max_chars=400, overlap=50)
        for idx, ch in enumerate(chunks):
            kb_docs.append({"id": f"{d['id']}_chunk_{idx}", "text": ch})
    return kb_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What is RAG?", help="Query to ask against the KB")
    parser.add_argument("--k", default=3, type=int, help="Top K retrieval")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help=(
            "Attempt to load a small local LLM and generate a short response "
            "(set this only if you have sufficient VRAM)."
        ),
    )
    parser.add_argument(
        "--force-device",
        choices=["cpu", "cuda"],
        help=(
            "Force a runtime device; use with care if your venv doesn't have "
            "CUDA-enabled PyTorch"
        ),
    )
    args = parser.parse_args()

    kb_docs = build_sample_kb()
    device = get_device()
    parser_force = getattr(args, "force_device", None)
    if parser_force:
        device = parser_force
        if parser_force == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    print("Warning: forced 'cuda' but torch lacks CUDA.")
                    print("Install CUDA-enabled PyTorch in your venv.")
                    device = "cpu"
            except Exception:
                print("Warning: unable to import torch; using CPU fallback.")
                device = "cpu"
    diags = cuda_diagnostics()
    print("Device diagnostics:", diags)

    retriever = Retriever(device=device)
    if device == "cpu":
        print("Note: your environment is CPU-only.")
    print("See README for instructions to enable GPU and CUDA-enabled")
    print("PyTorch.")
    retriever.build_index(kb_docs)
    results = retriever.retrieve(args.query, k=args.k)

    print("Query:", args.query)
    print("Top docs:")
    print(json.dumps(results, indent=2))

    # Build prompt from retrieved docs + query
    try:
        prompt = build_prompt(results, args.query)
    except Exception as e:
        print("Failed to build prompt from retrieved docs (using fallback):", e)
        prompt = (
            "Context:\n"
            + "\n\n".join([r.get("text", "") for r in results])
            + "\n\nQuestion: "
            + args.query
        )

    # If the user requested generation, use LocalLLM to produce an answer
    if args.with_llm:
        print("Attempting to generate an answer using LocalLLM")

        # Use a model path placeholder. Replace with a real model path on your machine.
        model_path = "models/your-model-here"

        # Instantiating LocalLLM; it tries to fallback to a small model if necessary
        try:
            llm = LocalLLM(model_path=model_path, device=device)
            response = llm.generate(prompt, max_new_tokens=128)
            print("\n\n=== ANSWER ===")
            print(response)
        except Exception as e:
            print("Failed to load or run LocalLLM:", e)


if __name__ == "__main__":
    main()