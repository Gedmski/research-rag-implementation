# Project Logs — Baseline RAG Implementation

**Date:** 2025-11-26

## Project Summary

This repository contains a research-focused, baseline Retrieval-Augmented Generation (RAG) implementation built in Python. The codebase demonstrates a simple pipeline that retrieves relevant context (documents/chunks) and uses a generative model to produce answers conditioned on retrieved passages. It is intended as a starting point for experimenting with RAG patterns, device-aware model execution (CPU/GPU), and small-scale retrieval components for research and prototyping.

## What the baseline provides

- **Core RAG code**: a `rag/` package with modular components for device management, retrieval, prompting, and generation.
- **Model assets**: a `models/gpt2/` folder with multiple model formats (TensorFlow, ONNX, TFLite, safetensors, etc.) to support different runtimes and experiments.
- **Scripts & notebooks**: convenience scripts and a Jupyter notebook demonstrating pipeline usage and quick experiments.
- **Tests**: basic unit tests under `tests/` to exercise core functionality.
- **Requirements**: `requirements.txt` and `requirements-gpu.txt` listing dependencies for CPU and GPU workflows.

## Files Added / Key Components

- `run_rag.py` — Example runner script to execute the RAG pipeline end-to-end.
- `free_rag_pipeline.ipynb` — Notebook demonstrating the pipeline and interactive experiments.
- `setup_gpu.py`, `diag_cuda.py` — GPU setup/diagnostic helpers.

- `rag/__init__.py` — Package init for the RAG components.
- `rag/device.py` — Device abstraction and utilities for selecting/executing on CPU vs GPU-backed runtimes.
- `rag/retriever.py` — Retrieval logic to find candidate documents or passages relevant to a query.
- `rag/prompting.py` — Prompt templates and helpers to convert retrieved context into model prompts.
- `rag/generation.py` — Generation orchestration that calls the model runtime with prompts and returned retrieved context.
- `rag/utils.py` — Utility functions used across the package.

- `tests/test_core.py`, `tests/test_device.py` — Unit tests validating basic pipeline pieces and device handling.

## Notable model assets

- `models/gpt2/` contains multiple artifacts to support different backends:
  - `model.safetensors`, `tf_model.h5`, `flax_model.msgpack` — model checkpoints in different frameworks
  - `onnx/` — ONNX exports (decoder models and merged variants)
  - `*.tflite` — TFLite versions for mobile/edge experiments
  - Tokenizer and config files for consistent tokenization across runtimes

## Changes Made (Baseline Implementation)

This baseline commit/implementation establishes the scaffolding and core features necessary for RAG experiments. In summary, the following high-level changes were made:

- Added a modular `rag/` package implementing device-aware generation, retrieval, and prompting.
- Included example runners and a notebook to show how to run the pipeline with the included model assets.
- Added multiple model formats under `models/gpt2/` so you can try different runtimes (TF, ONNX, TFLite, safetensors).
- Added simple unit tests under `tests/` to help validate device selection and core pipeline behavior.
- Added helper scripts for GPU setup and diagnostics to ease local experimentation on Windows (or other platforms).

These changes are additive: they introduce new modules, tests, and assets but do not modify external system configurations.

## How to run (quick)

1. Create and activate the appropriate Python environment.
2. Install dependencies (CPU or GPU):

```bash
python -m pip install -r requirements.txt
# or for GPU-enabled dependencies
python -m pip install -r requirements-gpu.txt
```

3. Run the example script:

```bash
python run_rag.py
```

Or open `free_rag_pipeline.ipynb` to walk through the pipeline interactively.

## Known limitations & next steps

- Retrieval is intentionally simple (baseline). Consider integrating a more advanced retriever (dense vectors, FAISS, ElasticSearch, etc.).
- Generation is written to work with multiple backends, but robust multi-backend CI/validation is needed.
- Add more comprehensive tests covering end-to-end behavior and failure modes.
- Add README sections showing recommended workflows and configuration tips for GPU execution.

## Contact / Authors

See `README.md` for more repository-level information and contact details.

---

Generated automatically: baseline logs summary describing the current repository contents and initial changes introduced for the RAG baseline implementation.
