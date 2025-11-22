from rag.device import cuda_diagnostics


def test_cuda_diagnostics_keys():
    diags = cuda_diagnostics()
    assert isinstance(diags, dict)
    # essential keys
    for k in ("torch_version", "cuda_version", "cuda_available"):
        assert k in diags
    # types
    assert isinstance(diags["cuda_available"], bool)
    assert isinstance(diags["torch_version"], str)
    # cuda_version can be None or str
    assert (
        diags["cuda_version"] is None
        or isinstance(diags["cuda_version"], str)
    )
