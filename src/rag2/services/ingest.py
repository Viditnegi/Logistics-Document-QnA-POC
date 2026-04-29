from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int
    section_title: str
    chunk_type: str  # "parent" or "child"
    parent_id: str | None = None
    parent: Any = None
    
    def metadata(self, document_id: str, filename: str) -> dict[str, Any]:
        return {
            "document_id": document_id,
            "filename": filename,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
            "parent_id": self.parent_id,
        }


def extract_elements(file_path: Path) -> list[Any]:
    from unstructured.partition.pdf import partition_pdf
    return partition_pdf(filename=str(file_path), infer_table_structure=True, strategy="hi_res")


def build_chunks(
    document_id: str,
    filename: str,
    elements: list[Any],
    chunk_size: int,
    chunk_overlap: int,
    file_path: Path | None = None,
) -> list[Chunk]:
    """Build parent + child chunks from PDF."""
    if file_path:
        parent_chunks = _extract_pages_as_parents(file_path)
    else:
        parent_chunks = _extract_elements_as_parents(elements)
    
    if not parent_chunks:
        parent_chunks = _extract_elements_as_parents(elements)
    
    child_chunks = _split_parents_into_children(parent_chunks, chunk_size, chunk_overlap)
    
    parent_map = {p.chunk_id: p for p in parent_chunks}
    for child in child_chunks:
        if child.parent_id and child.parent_id in parent_map:
            child.parent = parent_map[child.parent_id]
    
    return parent_chunks + child_chunks


def _extract_elements_as_parents(elements: list[Any]) -> list[Chunk]:
    """Fallback: extract from unstructured elements."""
    chunks = []
    current_section = "Document Overview"
    buffer = []
    buffer_page = 1
    
    for el in elements:
        el_type = el.category.lower()
        el_text = (el.text or "").strip()
        el_page = el.metadata.page_number if el.metadata and el.metadata.page_number else buffer_page
        
        if el_type == "title":
            if buffer:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text="\n\n".join(buffer),
                    page=buffer_page,
                    section_title=current_section,
                    chunk_type="parent",
                ))
                buffer.clear()
            current_section = el_text
            buffer_page = el_page
            continue
        
        if not el_text:
            continue
        
        buffer.append(el_text)
    
    if buffer:
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            text="\n\n".join(buffer),
            page=buffer_page,
            section_title=current_section,
            chunk_type="parent",
        ))
    
    return chunks


def _extract_pages_as_parents(file_path: Path | None) -> list[Chunk]:
    """Extract each page as a parent chunk using pdfplumber."""
    if not file_path:
        return []
    
    chunks = []
    footer_keywords = ["Page", "Powered by", "Demo"]
    skip_patterns = [
        "Carrier Rate and Load Confirmation", "Dispatcher", "Reference ID",
        "Phone", "Created On", "Shipping Date", "Booking Date", "Mailing Address",
    ]
    
    with pdfplumber.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            
            if page_num == 1:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=text.strip(),
                    page=page_num,
                    section_title="Document Overview",
                    chunk_type="parent",
                ))
            else:
                cleaned = _clean_page_text(text, skip_patterns, footer_keywords)
                if cleaned.strip():
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=cleaned.strip(),
                        page=page_num,
                        section_title="",
                        chunk_type="parent",
                    ))
    
    return chunks


def _clean_page_text(text: str, skip_patterns: list[str], footer_keywords: list[str]) -> str:
    lines = text.split("\n")
    cleaned = []
    header_set = set(h.strip() for h in lines[:8] if h.strip())
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line in header_set:
            continue
        if any(line.startswith(kw) for kw in footer_keywords):
            continue
        if any(skip in line for skip in skip_patterns):
            if any(line.startswith(skip) for skip in skip_patterns):
                continue
        cleaned.append(line)
    
    return "\n".join(cleaned)


def _split_parents_into_children(parent_chunks: list[Chunk], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split each parent into child chunks."""
    children = []
    
    for parent in parent_chunks:
        text = parent.text
        offset = 0
        
        while offset < len(text):
            chunk_text = text[offset:offset + chunk_size]
            children.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text.strip(),
                page=parent.page,
                section_title=parent.section_title,
                chunk_type="child",
                parent_id=parent.chunk_id,
            ))
            
            offset += chunk_size - chunk_overlap if chunk_overlap else chunk_size
            if chunk_overlap >= chunk_size:
                break
    
    return children


def _extract_elements_as_parents(elements: list[Any]) -> list[Chunk]:
    """Fallback: extract from unstructured elements."""
    chunks = []
    current_section = "Document Overview"
    buffer = []
    buffer_page = 1
    
    for el in elements:
        el_type = el.category.lower()
        el_text = (el.text or "").strip()
        el_page = el.metadata.page_number if el.metadata and el.metadata.page_number else buffer_page
        
        if el_type == "title":
            if buffer:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text="\n\n".join(buffer),
                    page=buffer_page,
                    section_title=current_section,
                    chunk_type="parent",
                ))
                buffer.clear()
            current_section = el_text
            buffer_page = el_page
            continue
        
        if not el_text:
            continue
        
        buffer.append(el_text)
    
    if buffer:
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            text="\n\n".join(buffer),
            page=buffer_page,
            section_title=current_section,
            chunk_type="parent",
        ))
    
    return chunks