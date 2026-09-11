from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.routes_analytics import router as analytics_router
from app.api.routes_review import queue_router, router as review_router
from app.api.schemas import DocumentStatusResponse, UploadResponse
from app.config import settings
from app.models.database import Document, DocumentStatus
from app.models.db import get_session, init_db
from app.tasks.pipeline_tasks import process_document

app = FastAPI(
    title="Multi-Modal Document Processor",
    description="OCR + LLM extraction + validation pipeline with human-in-the-loop review",
)

app.include_router(review_router)
app.include_router(queue_router)
app.include_router(analytics_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.processed_dir).mkdir(parents=True, exist_ok=True)


app.mount("/static/processed", StaticFiles(directory=settings.processed_dir), name="processed")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    source: str | None = Query(None, description="Vendor/source identifier for anomaly baselines"),
    session: Session = Depends(get_session),
) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    supported = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    if suffix not in supported:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Supported: {sorted(supported)}")

    document_id = str(uuid.uuid4())
    storage_path = Path(settings.upload_dir) / f"{document_id}{suffix}"
    with storage_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    document = Document(
        id=document_id,
        filename=file.filename or storage_path.name,
        storage_path=str(storage_path),
        mime_type=file.content_type or "application/octet-stream",
        source=source,
        status=DocumentStatus.UPLOADED,
    )
    session.add(document)
    session.commit()

    process_document.delay(document_id)

    return UploadResponse(document_id=document_id, filename=document.filename, status=document.status.value)


@app.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str, session: Session = Depends(get_session)) -> DocumentStatusResponse:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")

    return DocumentStatusResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status.value,
        document_type=document.document_type,
        error_message=document.error_message,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@app.get("/documents", response_model=list[DocumentStatusResponse])
def list_documents(
    status: str | None = Query(None),
    limit: int = Query(50, le=200),
    session: Session = Depends(get_session),
) -> list[DocumentStatusResponse]:
    query = select(Document).order_by(Document.created_at.desc()).limit(limit)
    if status:
        query = query.where(Document.status == status)

    documents = session.execute(query).scalars().all()
    return [
        DocumentStatusResponse(
            document_id=d.id,
            filename=d.filename,
            status=d.status.value,
            document_type=d.document_type,
            error_message=d.error_message,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        for d in documents
    ]
