"""Small script to print CUDA / device diagnostics for the RAG project."""
from rag.device import get_device, cuda_diagnostics


def main():
    device = get_device()
    print("Recommended device:", device)
    print("CUDA diagnostics:")
    for k, v in cuda_diagnostics().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
