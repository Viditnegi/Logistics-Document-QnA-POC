from typing import Literal

from pydantic import BaseModel, Field


class ShipmentData(BaseModel):
    shipment_id: str | None = None
    shipper: str | None = None
    consignee: str | None = None
    pickup_datetime: str | None = None
    delivery_datetime: str | None = None
    equipment_type: str | None = None
    mode: str | None = None
    rate: float | None = None
    currency: str | None = None
    weight: float | None = None
    carrier_name: str | None = None


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    section_count: int
    shipment_data: ShipmentData | None = None


class ChatRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=3)


class Citation(BaseModel):
    chunk_id: str
    page: int
    section_title: str
    chunk_type: Literal["section", "table", "paragraph", "composite", "parent", "child"]
    relevance: float


class RetrievedContext(BaseModel):
    label: str
    page: int
    section_title: str
    chunk_type: Literal["section", "table", "paragraph", "composite", "parent", "child"]
    relevance: float
    text: str

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    confidence_reason: str
    guardrail_status: Literal["ok", "blocked", "insufficient_evidence"]
    citations: list[Citation]
    contexts: list[RetrievedContext]
