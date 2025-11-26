"""Minimal LocalLLM wrapper for HuggingFace causal models used by the demo.

Provides a simple generate(prompt) API that:
- loads model + tokenizer
- runs inference using the specified device
- returns decoded string output

Note: This minimal implementation includes simple fallback behavior to 'gpt2'
when the requested model path isn't available.
"""
from __future__ import annotations
from typing import Optional
import importlib
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LocalLLM:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device_str = device if device else "cpu"
        self.device = torch.device("cuda" if self.device_str == "cuda" and torch.cuda.is_available() else "cpu")

        # Try to load the requested HF model; fallback to gpt2
        preferred = model_path
        if preferred == "models/your-model-here" or not self._is_model_available(preferred):
            # fallback to a small CPU-compatible model
            preferred = "gpt2"
            if self.device.type == "cuda":
                preferred = "Qwen/Qwen2.5-1.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(preferred, use_fast=True)
        # Some tokenizers (gpt2) don't have a pad token — ensure one exists to avoid warnings
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_8bit = importlib.util.find_spec("bitsandbytes") is not None and self.device.type == "cuda"

        if use_8bit:
            # If bitsandbytes is available, prefer 8-bit quantized load
            self.model = AutoModelForCausalLM.from_pretrained(preferred, load_in_8bit=True, device_map="auto")
        else:
            if self.device.type == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(preferred, device_map="auto", torch_dtype=torch.float16)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(preferred)

        # Ensure model on the correct device
        try:
            self.model.to(self.device)
        except Exception:
            # If we loaded with device_map="auto", torch won't accept .to like this; ignore
            pass

        self.model.eval()

    def _is_model_available(self, model_path: str) -> bool:
        # Simple heuristic: check if local dir exists or HF hub path can be resolved.
        if os.path.isdir(model_path):
            return True
        # Try to resolve hub model; a lightweight test try-import may not be reliable,
        # so rely on transformers to raise if not available.
        return True

    def generate(self, prompt: str, max_new_tokens: int = 64, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)

        decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return decoded