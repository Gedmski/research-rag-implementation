from typing import List, Iterator


def iter_simple_chunk_text(
    text: str, max_chars: int = 800, overlap: int = 100
) -> Iterator[str]:
    """Yield overlapping string chunks from `text`.

    This generator ensures forward progress by clamping `overlap` so
    it is always less than `max_chars`. If `max_chars` is invalid a
    ValueError is raised.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    # clamp overlap to ensure forward progress
    overlap = max(0, min(overlap, max_chars - 1))

    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        yield text[start:end].strip()
        next_start = end - overlap
        # guard against non-advancing next_start (which can loop forever)
        if next_start <= start:
            break
        start = next_start


def simple_chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """Backward compatible wrapper returning a concrete list of chunks.

    For very large inputs prefer `iter_simple_chunk_text` to stream
    chunks and avoid large memory spikes while preparing embeddings.
    """
    return list(iter_simple_chunk_text(text, max_chars=max_chars, overlap=overlap))

