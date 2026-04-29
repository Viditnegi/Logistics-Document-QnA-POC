from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag2.config import Settings, get_settings
from rag2.services.ingest import build_chunks, extract_elements


class RetrievalService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._vector_store: Chroma | None = None

    def ingest_pdf(self, document_id: str, filename: str, file_path: Path) -> tuple[int, int]:
        chunks = build_chunks(
            document_id=document_id,
            filename=filename,
            elements=extract_elements(file_path),
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            file_path=file_path,
        )
        documents = [Document(page_content=c.text, metadata=c.metadata(document_id, filename)) for c in chunks]
        if not documents:
            raise RuntimeError("No readable sections or tables were found in this PDF.")

        self._vector_store_or_raise().add_documents(documents=documents, ids=[c.chunk_id for c in chunks])
        section_count = len({d.metadata.get("section_title", "") for d in documents})
        return len(chunks), section_count

    def retrieve(self, document_id: str, question: str, limit: int | None = None) -> list[tuple[Document, float]]:
        results = self._vector_store_or_raise().similarity_search_with_score(
            question, k=limit or self.settings.retrieval_k, filter={"document_id": document_id}
        )
        
        normalized = []
        for doc, distance in results:
            relevance = 1.0 / (1.0 + float(distance))
            parent_id = doc.metadata.get("parent_id")
            
            if parent_id:
                try:
                    parent_results = self._vector_store_or_raise().get(where={"chunk_id": parent_id})
                    if parent_results.get("ids") and parent_results.get("documents"):
                        doc.page_content = parent_results["documents"][0]
                except Exception:
                    pass
            
            normalized.append((doc, round(relevance, 3)))
        return normalized

    def _vector_store_or_raise(self) -> Chroma:
        if self._vector_store is not None:
            return self._vector_store
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required before documents can be embedded or queried.")

        embeddings = OpenAIEmbeddings(model=self.settings.embedding_model, api_key=self.settings.openai_api_key)
        self._vector_store = Chroma(
            collection_name=self.settings.chroma_collection,
            persist_directory=str(self.settings.chroma_dir),
            embedding_function=embeddings,
        )
        return self._vector_store


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(get_settings())