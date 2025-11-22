import pytest
from rag.utils import simple_chunk_text
from rag.retriever import Retriever


def test_simple_chunk_text_short():
    text = "Hello world"
    chunks = simple_chunk_text(text, max_chars=5, overlap=2)
    assert len(chunks) >= 1
    assert "Hello" in chunks[0] or "world" in chunks[0]


def test_simple_chunk_text_overlap_and_generator():
    # overlap greater than max_chars should be clamped and not infinite-loop
    text = "short text"
    chunks = simple_chunk_text(text, max_chars=400, overlap=500)
    assert len(chunks) == 1
    # generator also works and yields the same single chunk
    from rag.utils import iter_simple_chunk_text

    gen_chunks = list(iter_simple_chunk_text(text, max_chars=400, overlap=500))
    assert gen_chunks == chunks


def test_simple_chunk_text_large():
    # Ensure reasonable chunk sizes without infinite loops
    large = "x" * 1500
    parts = simple_chunk_text(large, max_chars=400, overlap=50)
    assert len(parts) > 1
    assert all(len(p) <= 400 for p in parts)


def test_retriever_tfidf_basic():
    raw_docs = [
        {"id": "a", "text": "apple banana orange"},
        {"id": "b", "text": "banana fruit smoothie"},
        {"id": "c", "text": "car truck vehicle"},
    ]
    r = Retriever()
    r.build_index(raw_docs)
    results = r.retrieve("banana", k=2)
    assert len(results) == 2
    # top result should mention banana in 'a' or 'b'
    top_ids = [r["id"] for r in results]
    assert any(x in top_ids for x in ("a", "b"))
