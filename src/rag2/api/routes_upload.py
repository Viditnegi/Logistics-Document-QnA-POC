from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from rag2.config import get_settings
from rag2.models import UploadResponse
from rag2.services.extraction import get_extraction_service
from rag2.services.retrieval import get_retrieval_service


router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_file_type",
                "message": "Only PDF uploads are supported in this POC.",
                "where": "upload",
            },
        )

    settings = get_settings()
    document_id = uuid4().hex
    safe_name = Path(file.filename or "document.pdf").name
    destination = settings.upload_dir / f"{document_id}-{safe_name}"
    payload = await file.read()
    destination.write_bytes(payload)

    try:
        chunk_count, section_count = get_retrieval_service().ingest_pdf(
            document_id=document_id,
            filename=safe_name,
            file_path=destination,
        )
    except RuntimeError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "runtime_error",
                "message": str(exc),
                "where": "upload",
            },
        ) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": type(exc).__name__,
                "message": str(exc),
                "where": "upload",
            },
        ) from exc

    shipment_data = None
    try:
        shipment_data = get_extraction_service().extract_shipment_data(document_id)
    except Exception:
        pass

    return UploadResponse(
        document_id=document_id,
        filename=safe_name,
        chunk_count=chunk_count,
        section_count=section_count,
        shipment_data=shipment_data,
    )
