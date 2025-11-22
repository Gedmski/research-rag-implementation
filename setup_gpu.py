"""Helper to print suggested install commands for enabling GPU in this venv.

This script does not install anything automatically unless `--apply` is passed
and you confirm; it's intended to be a guided helper to fix CPU-only PyTorch
installs in a venv.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import platform
from typing import Optional


def has_nvidia_smi() -> bool:
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False


def get_nvidia_cuda_version() -> Optional[str]:
    """Return the CUDA version reported by nvidia-smi, if available."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=cuda_version",
                "--format=csv,noheader",
            ],
            stderr=subprocess.STDOUT,
        )
        s = out.decode("utf-8").strip()
        if s:
            # nvidia-smi may return multiple lines; take the first
            first = s.splitlines()[0].strip()
            return first
    except Exception:
        pass
    return None


def torch_cuda_available(info: bool = False) -> Optional[bool]:
    try:
        import torch

        if info:
            return (
                torch.cuda.is_available(),
                torch.__version__,
                torch.version.cuda,
            )
        return torch.cuda.is_available()
    except Exception:
        return None


def suggest_commands(cuda_version: str = "11.8") -> None:
    print(
        "Suggested commands to enable GPU in your active venv (Windows/CMD):",
    )
    print("# 1. (Optional) uninstall existing torch to avoid conflicts:")
    print("python -m pip uninstall -y torch torchvision torchaudio")
    print(
        "python -m pip install torch torchvision torchaudio --index-url",
        f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}",
    )
    print()
    print("# 3. Optional: install faiss-gpu via conda (preferred)")
    print("#    or fallback to faiss-cpu via pip")
    print("# (concrete example using conda; ensures binary compatibility):")
    print(f"conda install -c pytorch faiss-gpu cudatoolkit={cuda_version}")
    print()
    print("# 4. Optional: install model quantization helpers for LLMs:")
    print("python -m pip install bitsandbytes accelerate transformers")


def prompt_confirm() -> bool:
    val = (
        input(
            "Proceed to run suggested commands in the current venv? (yes/no): "
        )
        .strip()
        .lower()
    )
    return val in ("y", "yes")


def run_cmd(cmd: str) -> None:
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply suggested commands automatically",
    )
    parser.add_argument(
        "--cuda-version",
        default="11.8",
        help="CUDA version to install for PyTorch/faiss",
    )
    args = parser.parse_args()

    print("System:", platform.platform())
    print("nvidia-smi present:", has_nvidia_smi())
    tinfo = torch_cuda_available(info=True)
    print("Current torch cuda state:", tinfo)

    detected = get_nvidia_cuda_version()
    if detected:
        print("nvidia-smi reports CUDA version:", detected)
        # Suggest wheel index based on detected version; use detected as-is
        suggest_commands(detected)
    else:
        suggest_commands(args.cuda_version)

    if args.apply:
        if not prompt_confirm():
            print("Aborting: user did not confirm.")
            sys.exit(0)
        print(
            "Applying commands (non-destructive). If any command fails, fix",
        )
        print("manually.")
        try:
            run_cmd("python -m pip uninstall -y torch torchvision torchaudio")
            cu_str = args.cuda_version.replace('.', '')
            cmd = (
                "python -m pip install torch torchvision "
                "torchaudio --index-url "
                + f"https://download.pytorch.org/whl/cu{cu_str}"
            )
            run_cmd(cmd)
            print("Please use conda to install `faiss-gpu` if desired.")
            print("This is recommended to ensure binary compatibility.")
        except Exception as e:
            print("One of the commands failed:", e)
            print(
                "Please run the above suggested commands manually and confirm",
            )
            print("compatibility.")


if __name__ == "__main__":
    main()
