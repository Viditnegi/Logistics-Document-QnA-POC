from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rag2.models import ChatRequest, ChatResponse
from rag2.services.qa import get_qa_service


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return get_qa_service().answer_question(request.document_id, request.question)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "runtime_error",
                "message": str(exc),
                "where": "chat",
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": type(exc).__name__,
                "message": str(exc),
                "where": "chat",
            },
        ) from exc
