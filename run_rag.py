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

    if args.with_llm:
        # Try to load a small HF model locally and generate a short answer
        from transformers import AutoTokenizer, AutoModelForCausalLM

        MODEL_NAME = (
            "gpt2" if device == "cpu" else "Qwen/Qwen2.5-1.5B-Instruct"
        )
        print("Attempting to load LLM:", MODEL_NAME)
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if device == "cpu":
                model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            else:
                # prefer device_map to allow automatic placement for big models
                # If bitsandbytes is present, prefer 8-bit quantized load
                use_8bit = False
                try:
                    import importlib

                    use_8bit = (
                        importlib.util.find_spec("bitsandbytes") is not None
                    )
                except Exception:
                    use_8bit = False
                if use_8bit:
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME,
                        load_in_8bit=True,
                        device_map="auto",
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME, device_map="auto", torch_dtype="auto"
                    )
            model.eval()
            prompt = (
                "Use the following context to answer: "
                + results[0]["text"]
                + "\nQuestion: "
                + args.query
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == "cuda":
                inputs = inputs.to("cuda")
            import torch
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64)
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            print("LLM output (first 512 chars):\n", decoded[:512])
        except Exception as e:
            print("Failed to load or run LLM locally:", e)


if __name__ == "__main__":
    main()
