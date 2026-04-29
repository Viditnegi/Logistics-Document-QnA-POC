from types import SimpleNamespace
from rag2.services.ingest import build_chunks

def test_chunker_keeps_tables_separate_from_paragraphs() -> None:
    elements = [
        SimpleNamespace(category="Title", text="1. Charges", metadata=SimpleNamespace(page_number=1)),
        SimpleNamespace(category="NarrativeText", text="Fuel surcharge applies.", metadata=SimpleNamespace(page_number=1)),
        SimpleNamespace(category="Table", text="Lane | Price\nLAX-DAL | 1200", metadata=SimpleNamespace(page_number=1, text_as_html="<table>...</table>")),
    ]

    chunks = build_chunks(
        document_id="doc-1",
        filename="rates.pdf",
        elements=elements,
        chunk_size=1000,
        chunk_overlap=100,
    )

    children = [c for c in chunks if c.chunk_type == "child"]
    assert len(children) > 0
    for child in children:
        assert child.parent_id is not None


def test_chunker_rolls_long_sections_into_multiple_chunks() -> None:
    long_text = "Word " * 200 
    elements = [
        SimpleNamespace(category="Title", text="2. Terms", metadata=SimpleNamespace(page_number=1)),
        SimpleNamespace(category="NarrativeText", text=long_text, metadata=SimpleNamespace(page_number=1)),
    ]

    chunks = build_chunks(
        document_id="doc-1",
        filename="terms.pdf",
        elements=elements,
        chunk_size=500,
        chunk_overlap=50,
    )

    children = [c for c in chunks if c.chunk_type == "child"]
    assert len(children) > 1
    assert all(c.section_title == "2. Terms" for c in children)