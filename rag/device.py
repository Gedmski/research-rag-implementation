"""Device detection and CUDA diagnostics for the RAG pipeline."""
from __future__ import annotations
import torch


def get_device() -> str:
    """Return a recommended device string ('cuda' or 'cpu')."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def cuda_diagnostics() -> dict:
    """Return a small diagnostics dict about the CUDA environment.

    Useful to print in notebooks/CI to confirm GPU availability and properties.
    """
    info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        # Use only device 0 for diagnostics
        dev_props = torch.cuda.get_device_properties(0)
        info.update({
            "device_name": dev_props.name,
            "total_memory": dev_props.total_memory,
            "major": dev_props.major,
            "minor": dev_props.minor,
        })
    return info
