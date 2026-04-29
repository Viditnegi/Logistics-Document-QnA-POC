from langchain_core.documents import Document

from rag2.config import Settings
from rag2.services.qa import QAService


class FakeRetrievalService:
    def retrieve(self, document_id: str, question: str, limit: int | None = None):
        return [
            (
                Document(
                    page_content="Detention charges apply after 48 hours. The fee is USD 75 per hour.",
                    metadata={
                        "page": 3,
                        "section_title": "Accessorial Charges",
                        "chunk_type": "composite",  # Changed from paragraph
                        "chunk_id": "test-uuid-123", # Changed from source_span
                        "document_id": document_id,
                        "filename": "test.pdf"
                    },
                ),
                0.82,
            )
        ]

class FailingLLM:
    def invoke(self, prompt: str):
        raise RuntimeError("429 RESOURCE_EXHAUSTED test quota failure")


class FakeQAService(QAService):
    def _llm_or_raise(self):
        return FailingLLM()


def test_qa_falls_back_to_extractive_answer_when_llm_is_unavailable() -> None:
    service = FakeQAService(Settings(openai_api_key="test-key"), FakeRetrievalService())

    response = service.answer_question("doc-1", "What are detention charges after 48 hours?")

    assert response.guardrail_status == "ok"
    assert "Model generation failed:" in response.answer
    assert "429 RESOURCE_EXHAUSTED test quota failure" in response.answer
    assert "most relevant retrieved evidence" in response.answer
    assert "USD 75 per hour" in response.answer
    assert response.citations[0].page == 3
    assert response.contexts[0].label == "C1"
    assert response.contexts[0].text == "Detention charges apply after 48 hours. The fee is USD 75 per hour."
