# Fully-Free RAG Pipeline — Local Notebook + Script

This repository contains a ready-to-run RAG (Retrieval-Augmented Generation) pipeline blueprint and a small demo runner. It follows the `CONTEXT.MD` blueprint and provides a Jupyter notebook structure, helper modules, a CLI runner script, and tests. Everything is designed to be runnable locally and free — no OpenAI or paid APIs.

Quick start
1. (Optional) Create and activate a Python virtual environment.
2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```
3. Open the Jupyter notebook `free_rag_pipeline.ipynb` and run cells top-to-bottom.

Or run the demo CLI (toy demo that does not require GPU or large models):
```bash
python run_rag.py
```

CLI usage notes
- Use `--force-device cpu` or `--force-device cuda` to force device selection.
	- `--force-device cuda` will try to run on GPU but will fall back to CPU if your venv lacks CUDA-enabled PyTorch.
- Use `--with-llm` to attempt to load a small local LLM for generation; omit if you have limited VRAM.

Files included
- `free_rag_pipeline.ipynb`: Notebook blueprint with cell-by-cell content.
- `run_rag.py`: Minimal CLI to run a toy demo using small or mocked embeddings.
- `rag/`: Python package with helper functions used by the notebook/CLI.
- `tests/`: Unit tests for the chunking and retrieval utilities.
- `requirements.txt`: Required packages.

Notes
- If you run the notebook and want to use real models, make sure you have GPU drivers and enough VRAM or choose smaller models / quantization.
- The CLI fallback will use numpy-based embeddings if `sentence-transformers`, `faiss`, or model downloads are not available.

Using a GPU (CUDA) — quick guide
1. Verify your NVIDIA drivers are current and compatible with a CUDA toolkit. RTX 4xxx cards usually need recent drivers.
2. Install a CUDA-enabled PyTorch build. For example (on Windows, choose the right CUDA version for your system):
```cmd
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118  # or cu121 etc. Pick matching CUDA
```
3. Optionally install FAISS GPU build if you want GPU-backed FAISS accelerated searches
	(this is easiest via conda; pip may not provide a GPU wheel on all platforms):
```cmd
conda install -c pytorch faiss-gpu cudatoolkit=11.8
```
4. If using LLMs, consider `bitsandbytes` for 8-bit loading and `accelerate` for device_map automation:
```cmd
pip install bitsandbytes accelerate
```
5. Re-open `free_rag_pipeline.ipynb` and re-run cells — the notebook will print device diagnostics and will use GPU for embedding operations if available.

If you have CUDA installed and want to use a GPU-optimized environment, you can use the GPU requirement file:
```cmd
pip install -r requirements-gpu.txt
```
Note: For `torch` with CUDA, prefer following the official install commands for your CUDA version.
Check your GPU is visible
1. Run `nvidia-smi` in a shell. If this prints device information, your Nvidia drivers are installed and the GPU is available.
2. Confirm PyTorch with CUDA is installed by starting Python and running:
```py
import torch
print(torch.cuda.is_available(), torch.__version__, torch.version.cuda)
```
This should print `(True, '<torch-version>', '<cuda-version>')` if CUDA-enabled PyTorch is present.

To get suggested install commands for your active venv, run the helper:
```cmd
python setup_gpu.py
```
Add `--apply` to attempt reinstallation in your active venv:
```cmd
python setup_gpu.py --apply
```
This will print suggested pip/conda commands; if `--apply` is used it will run `pip uninstall` and `pip install` commands in the current venv.

Install CUDA-enabled PyTorch on Windows (example commands)
1) For pip-install with CUDA 11.8 (choose the CUDA matching your driver):
```cmd
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
2) For conda (recommended for FAISS GPU on Windows):
```cmd
conda install pytorch torchvision --channel pytorch --yes
conda install -c pytorch faiss-gpu cudatoolkit=11.8  # pick a compatible cudatoolkit
```

If you cannot install FAISS with GPU on Windows, you can still run `faiss-cpu` or continue with the TF-IDF fallback.

Notes on model loading and VRAM
- If you plan to load a language model, prefer small ones or use quantization (bitsandbytes) to reduce VRAM usage.
- For HF `transformers`: `device_map="auto"` with `accelerate` installed helps place the model on GPU.
 - When indexing large document sets, prefer streaming chunks into your vector DB instead of collecting them all in memory. Use `iter_simple_chunk_text` from `rag.utils` to yield chunks one-by-one.

License
This project is provided as-is (no license specified).
