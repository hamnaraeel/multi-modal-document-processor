from app.extraction.chunking import merge_chunk_extractions, split_into_chunks
from app.extraction.schemas import Invoice, LineItem


def test_split_into_chunks_respects_char_budget():
    pages = [(1, "a" * 3000), (2, "b" * 3000), (3, "c" * 3000)]
    chunks = split_into_chunks(pages, max_chars=8000)
    assert len(chunks) == 2
    assert chunks[0].page_numbers == [1, 2]
    assert chunks[1].page_numbers == [3]


def test_split_into_chunks_never_exceeds_budget_per_chunk():
    pages = [(i, "x" * 5000) for i in range(1, 4)]
    chunks = split_into_chunks(pages, max_chars=8000)
    assert len(chunks) == 3
    assert all(len(c.text) <= 8000 for c in chunks)


def test_merge_single_chunk_returns_it_unchanged():
    invoice = Invoice(vendor_name="Acme", total_amount=100.0)
    merged, conflicts = merge_chunk_extractions(Invoice, [(0, invoice)])
    assert merged is invoice
    assert conflicts == []


def test_merge_concatenates_list_fields_and_dedupes():
    a = Invoice(vendor_name="Acme", line_items=[LineItem(description="Widget A", total=10)])
    b = Invoice(vendor_name="Acme", line_items=[LineItem(description="Widget A", total=10), LineItem(description="Widget B", total=20)])
    merged, _ = merge_chunk_extractions(Invoice, [(0, a), (1, b)])
    assert len(merged.line_items) == 2


def test_merge_flags_scalar_conflicts():
    a = Invoice(vendor_name="Acme", total_amount=100.0)
    b = Invoice(vendor_name="Acme Corp", total_amount=100.0)
    merged, conflicts = merge_chunk_extractions(Invoice, [(0, a), (1, b)])
    assert any(c.field == "vendor_name" for c in conflicts)
