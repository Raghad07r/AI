"""
tests/checks.py
Validation functions for the project notebook exercises.
Mirrors the check pattern used in the lab notebooks (checks.check_lab_X_Y).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Exercise 1 — Ingestion: Text file -> chunks -> vector store
# ---------------------------------------------------------------------------

def check_project_ex1(doc, chunks, store) -> None:
    assert doc is not None, "doc is None — did you call extract_document()?"
    assert hasattr(doc, "content"), "doc must be a Document dataclass"
    assert doc.doc_type == "text", f"Expected doc_type='text', got '{doc.doc_type}'"
    assert doc.word_count > 0, "word_count should be > 0 after extraction"

    assert chunks is not None and len(chunks) > 0, \
        "chunks is empty — did you call chunker.chunk_document()?"
    for c in chunks:
        assert "text" in c and "metadata" in c, \
            "Each chunk must have 'text' and 'metadata' keys"
        assert "embedding" in c, \
            "Each chunk must have an 'embedding' key — did you attach embeddings?"
        assert isinstance(c["embedding"], list) and len(c["embedding"]) > 0, \
            "embedding must be a non-empty list of floats"

    count_before = store.get_stats()["count"]
    store.add_documents(chunks)
    count_after = store.get_stats()["count"]
    assert count_after > count_before, \
        "Collection count did not increase after add_documents()"

    print(f"✓ Exercise 1 passed — {len(chunks)} chunks ingested, "
          f"store now has {count_after} documents.")


# ---------------------------------------------------------------------------
# Exercise 2 — Retrieval: Metadata filtering
# ---------------------------------------------------------------------------

def check_project_ex2(pdf_results) -> None:
    assert pdf_results is not None, \
        "pdf_results is None — did you call store.search()?"
    assert len(pdf_results) > 0, \
        "No results returned — verify filter_conditions and that PDF docs exist in the store"
    for r in pdf_results:
        assert "score" in r and "text" in r and "metadata" in r, \
            "Each result must have 'score', 'text', and 'metadata' keys"
        assert r["metadata"].get("doc_type") == "pdf", \
            f"Result has doc_type='{r['metadata'].get('doc_type')}', expected 'pdf'"
    print(f"✓ Exercise 2 passed — {len(pdf_results)} PDF-only results returned.")


# ---------------------------------------------------------------------------
# Exercise 3 — Observability: Loop detection
# ---------------------------------------------------------------------------

def check_project_ex3(LoopDetectorClass, LoopAwareRAGServiceClass, store, tracer, model_id) -> None:
    # Verify LoopDetector logic
    det = LoopDetectorClass(threshold=2)
    assert det.check("test", "q") is False, "First call should not be a loop"
    assert det.check("test", "q") is False, "Second call (count==1) should not be a loop"
    assert det.check("test", "q") is True,  "Third call (count==2) should be a loop"

    # Verify LoopAwareRAGService stops after threshold
    svc = LoopAwareRAGServiceClass(
        vector_store=store, tracer=tracer, model_id=model_id, loop_threshold=2
    )
    assert svc.loop_detector is not None, \
        "loop_detector is None — did you initialize LoopDetector in __init__?"

    q = "What is cosine similarity? (loop test)"
    r1 = svc.answer(q)
    r2 = svc.answer(q)
    r3 = svc.answer(q)  # should trigger loop guard

    assert r3 is not None, "answer() returned None on loop"
    assert "loop" in r3.get("answer", "").lower() or "repeated" in r3.get("answer", "").lower(), \
        f"Expected loop warning message, got: {r3.get('answer', '')[:100]}"

    print("✓ Exercise 3 passed — LoopDetector correctly halts repeated queries.")


# ---------------------------------------------------------------------------
# Exercise 4 — Cost estimation: plan_ingestion_cost()
# ---------------------------------------------------------------------------

def check_project_ex4(plan: dict) -> None:
    required_keys = {"total_documents", "total_chunks", "total_tokens", "cost_usd", "model"}
    missing = required_keys - set(plan.keys())
    assert not missing, f"plan is missing keys: {missing}"

    assert plan["total_documents"] > 0, "total_documents must be > 0"
    assert plan["total_chunks"] >= plan["total_documents"], \
        "total_chunks should be >= total_documents (at least 1 chunk per doc)"
    assert plan["total_tokens"] > 0, "total_tokens must be > 0"
    assert plan["cost_usd"] is not None and plan["cost_usd"] > 0, \
        "cost_usd must be a positive float"
    assert isinstance(plan["model"], str) and "embedding" in plan["model"].lower(), \
        "model should be an embedding model name"

    print(f"✓ Exercise 4 passed — Plan: {plan['total_chunks']} chunks, "
          f"{plan['total_tokens']:,} tokens, ${plan['cost_usd']:.6f} USD.")
