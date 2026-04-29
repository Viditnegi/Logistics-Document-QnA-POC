from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from rag2.config import Settings, get_settings
from rag2.models import ShipmentData
from rag2.services.retrieval import RetrievalService, get_retrieval_service


EXTRACTION_PROMPT = """You are a logistics document parser specialized in extracting shipment details from bills of lading, carrier confirmations, and shipping documents.

Extract the following fields from the document text below. If a field is not found or unclear, return null for that field.

Fields to extract:
1. shipment_id - The unique shipment/reference ID (like BOL number, tracking number, reference ID)
2. shipper - The company or person shipping the goods (sender)
3. consignee - The company or person receiving the goods (receiver)
4. pickup_datetime - When the shipment was picked up or scheduled for pickup
5. delivery_datetime - When the shipment was delivered or scheduled for delivery
6. equipment_type - Type of equipment: container size (20ft, 40ft, 45ft), trailer type, van, etc.
7. mode - Transportation mode: FTL, LTL, air, ocean, rail, ground, etc.
8. rate - The shipping rate/cost (numeric value only, no currency symbol)
9. currency - The currency for the rate: USD, CAD, etc.
10. weight - Weight of the shipment in lbs or kgs (numeric value only)
11. carrier_name - The carrier company's name

Return ONLY a valid JSON object with these exact field names. No explanation or additional text.

Document text:
{document_text}
"""


class ExtractionService:
    def __init__(self, settings: Settings, retrieval: RetrievalService):
        self.settings = settings
        self.retrieval = retrieval
        self._llm: ChatOpenAI | None = None

    def extract_shipment_data(self, document_id: str) -> ShipmentData | None:
        hits = self.retrieval.retrieve(document_id=document_id, question="shipment details shipper consignee pickup delivery rate weight", limit=3)
        
        if not hits:
            return None
        
        doc_text = "\n\n".join(d.page_content[:2000] for d, _ in hits)
        
        try:
            response = self._llm_or_raise().invoke(EXTRACTION_PROMPT.format(document_text=doc_text))
            content = str(response.content).strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = ShipmentData.model_validate_json(content)
            return data
        except ValidationError:
            return None
        except Exception:
            return None

    def _llm_or_raise(self) -> ChatOpenAI:
        if self._llm is not None:
            return self._llm
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for extraction.")
        self._llm = ChatOpenAI(model=self.settings.chat_model, api_key=self.settings.openai_api_key, temperature=0.1)
        return self._llm


@lru_cache(maxsize=1)
def get_extraction_service() -> ExtractionService:
    return ExtractionService(get_settings(), get_retrieval_service())