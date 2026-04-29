from __future__ import annotations

from functools import lru_cache
import re

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from rag2.config import Settings, get_settings
from rag2.models import ChatResponse, Citation, RetrievedContext
from rag2.services.guardrails import evidence_guardrail, preflight_guardrail
from rag2.services.retrieval import RetrievalService, get_retrieval_service
from rag2.services.scoring import confidence_score


ANSWER_PROMPT = """You are a transportation management system assistant.
Answer the question using only the provided context.
If the context is missing the answer, say that there is not enough evidence in the document.
Mention the relevant citation labels in square brackets such as [C1].

Question:
{question}

Context:
{context}
"""


class QAService:
    def __init__(self, settings: Settings, retrieval: RetrievalService):
        self.settings = settings
        self.retrieval = retrieval
        self._llm: ChatOpenAI | None = None

    def answer_question(self, document_id: str, question: str) -> ChatResponse:
        preflight = preflight_guardrail(question)
        if preflight.status != "ok":
            return _response(preflight.reason, 0.0, "The request was blocked before retrieval.", "blocked", [])

        hits = self.retrieval.retrieve(document_id=document_id, question=question)
        evidence = evidence_guardrail(hits[0][1] if hits else 0.0, len(hits), self.settings.confidence_threshold)

        if evidence.status != "ok":
            confidence, reason = _score(hits, evidence.status)
            return _response(
                "I do not have enough grounded evidence in the uploaded document to answer that reliably.",
                confidence,
                reason,
                "insufficient_evidence",
                hits,
            )

        prompt = ANSWER_PROMPT.format(question=question, context=_prompt_context(hits))
        try:
            response = self._llm_or_raise().invoke(prompt)
            answer = str(response.content).strip()
        except Exception as exc:
            answer = _extractive_answer(question, hits, f"Model generation failed: {_error_summary(exc)}")

        confidence, reason = _score(hits, "ok")
        return _response(answer, confidence, reason, "ok", hits)

    def _llm_or_raise(self) -> ChatOpenAI:
        if self._llm is not None:
            return self._llm
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required before asking questions.")
        self._llm = ChatOpenAI(model=self.settings.chat_model, api_key=self.settings.openai_api_key, temperature=0.1)
        return self._llm


def _response(answer: str, confidence: float, reason: str, status: str, hits: list[tuple[Document, float]]) -> ChatResponse:
    top_hits = hits[:4]
    return ChatResponse(
        answer=answer,
        confidence=confidence,
        confidence_reason=reason,
        guardrail_status=status,
        citations=[_citation(doc, rel) for doc, rel in top_hits],
        contexts=[_context(i, doc, rel) for i, (doc, rel) in enumerate(top_hits, 1)],
    )


def _score(hits: list[tuple[Document, float]], guardrail_status: str) -> tuple[float, str]:
    relevances = [r for _, r in hits]
    return confidence_score(relevances, min(len(hits), 4), guardrail_status)


def _prompt_context(hits: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[C{i}] page={d.metadata.get('page')} section={d.metadata.get('section_title')} "
        f"type={d.metadata.get('chunk_type')} relevance={r}\n{d.page_content}"
        for i, (d, r) in enumerate(hits, 1)
    )


def _extractive_answer(question: str, hits: list[tuple[Document, float]], header: str) -> str:
    terms = {t for t in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(t) > 3}
    snippets = []
    for i, (doc, _) in enumerate(hits[:3], 1):
        snippet = _best_snippet(doc.page_content, terms)
        page = doc.metadata.get("page", "unknown")
        section = doc.metadata.get("section_title", "Document Overview")
        snippets.append(f"[C{i}] Page {page}, {section}: {snippet}")
    return "\n\n".join([f"{header}\n\nHere is the most relevant retrieved evidence instead:", *snippets])


def _best_snippet(text: str, question_terms: set[str]) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "No readable text was found in this chunk."
    ranked = sorted(lines, key=lambda l: sum(t in l.lower() for t in question_terms), reverse=True)
    snippet = " ".join(ranked[:3])
    return snippet[:800] + ("..." if len(snippet) > 800 else "")


def _error_summary(exc: Exception, limit: int = 1200) -> str:
    msg = re.sub(r"\s+", " ", str(exc)).strip() or "No error message was returned."
    summary = f"{type(exc).__name__}: {msg}"
    return summary[:limit] + ("..." if len(summary) > limit else "")


def _citation(doc: Document, relevance: float) -> Citation:
    return Citation(
        chunk_id=doc.metadata.get("chunk_id", "unknown"),
        page=int(doc.metadata.get("page", 0)),
        section_title=str(doc.metadata.get("section_title", "Document Overview")),
        chunk_type=doc.metadata.get("chunk_type", "composite"),
        relevance=relevance,
    )


def _context(index: int, doc: Document, relevance: float) -> RetrievedContext:
    chunk_type = "parent" if doc.metadata.get("parent_id") else doc.metadata.get("chunk_type", "composite")
    return RetrievedContext(
        label=f"C{index}",
        page=int(doc.metadata.get("page", 0)),
        section_title=str(doc.metadata.get("section_title", "Document Overview")),
        chunk_type=chunk_type,
        relevance=relevance,
        text=_trim(doc.page_content),
    )


def _trim(text: str, limit: int = 1600) -> str:
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    return normalized[:limit] + ("..." if len(normalized) > limit else "")


@lru_cache(maxsize=1)
def get_qa_service() -> QAService:
    return QAService(get_settings(), get_retrieval_service())