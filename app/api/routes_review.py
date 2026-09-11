"""Endpoints backing the human review interface: fetching documents that
need review (single + prioritized queue), rendering their extracted data
side-by-side with page images, inline field corrections, and batch
approve/reject/edit decisions.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import CorrectionRequest, DocumentDetailResponse, ReviewDecisionRequest
from app.models.database import (
    Correction,
    Document,
    DocumentPage,
    DocumentStatus,
    ExtractionResult,
    ReviewSession,
    ValidationResult,
)
from app.models.db import get_session

router = APIRouter(prefix="/documents", tags=["review"])
queue_router = APIRouter(prefix="/review", tags=["review"])

REVIEW_STATUSES = {DocumentStatus.PENDING_FAST_REVIEW, DocumentStatus.PENDING_DETAILED_REVIEW}


class QueueItem(BaseModel):
    document_id: str
    filename: str
    status: str
    document_type: str | None
    overall_confidence: float | None
    routing_reason: str | None


@queue_router.get("/queue", response_model=list[QueueItem])
def get_review_queue(session: Session = Depends(get_session)) -> list[QueueItem]:
    """Prioritized: detailed-review first, then lowest confidence first --
    the documents most likely to need real attention surface at the top."""
    query = (
        select(Document, ExtractionResult, ValidationResult)
        .join(ExtractionResult, ExtractionResult.document_id == Document.id, isouter=True)
        .join(ValidationResult, ValidationResult.document_id == Document.id, isouter=True)
        .where(Document.status.in_(REVIEW_STATUSES))
    )
    rows = session.execute(query).all()

    items = [
        QueueItem(
            document_id=doc.id,
            filename=doc.filename,
            status=doc.status.value,
            document_type=doc.document_type,
            overall_confidence=extraction.overall_confidence if extraction else None,
            routing_reason=validation.routing_reason if validation else None,
        )
        for doc, extraction, validation in rows
    ]

    items.sort(
        key=lambda i: (
            0 if i.status == DocumentStatus.PENDING_DETAILED_REVIEW.value else 1,
            i.overall_confidence if i.overall_confidence is not None else 0.0,
        )
    )
    return items


@router.get("/{document_id}/detail", response_model=DocumentDetailResponse)
def get_document_detail(document_id: str, session: Session = Depends(get_session)) -> DocumentDetailResponse:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")

    extraction = session.execute(
        select(ExtractionResult).where(ExtractionResult.document_id == document_id)
    ).scalar_one_or_none()
    validation = session.execute(
        select(ValidationResult).where(ValidationResult.document_id == document_id)
    ).scalar_one_or_none()
    pages = session.execute(
        select(DocumentPage)
        .where(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number)
    ).scalars().all()

    return DocumentDetailResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status.value,
        document_type=document.document_type,
        page_image_urls=[f"/static/processed/{document.id}/page_{p.page_number}.png" for p in pages],
        fields=extraction.fields if extraction else None,
        field_confidence=extraction.field_confidence if extraction else None,
        field_sources=extraction.field_sources if extraction else None,
        conflicts=extraction.conflicts if extraction else None,
        overall_confidence=extraction.overall_confidence if extraction else None,
        type_errors=validation.type_errors if validation else None,
        business_rule_warnings=validation.business_rule_warnings if validation else None,
        business_rule_failures=validation.business_rule_failures if validation else None,
        anomalies=validation.anomalies if validation else None,
        routing_decision=validation.routing_decision if validation else None,
        routing_reason=validation.routing_reason if validation else None,
    )


@router.post("/{document_id}/corrections")
def submit_correction(
    document_id: str, correction: CorrectionRequest, session: Session = Depends(get_session)
) -> dict:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")

    db_correction = Correction(document_id=document_id, **correction.model_dump())
    session.add(db_correction)

    extraction = session.execute(
        select(ExtractionResult).where(ExtractionResult.document_id == document_id)
    ).scalar_one_or_none()
    if extraction and correction.field_name in extraction.fields:
        extraction.fields[correction.field_name] = correction.corrected_value
        # JSON columns need reassignment (not in-place mutation) to be tracked by SQLAlchemy.
        extraction.fields = dict(extraction.fields)

    session.commit()
    return {"status": "recorded", "correction_id": db_correction.id}


@router.post("/{document_id}/review")
def submit_review_decision(
    document_id: str, decision: ReviewDecisionRequest, session: Session = Depends(get_session)
) -> dict:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")

    now = datetime.utcnow()
    review_session = ReviewSession(
        document_id=document_id,
        reviewer=decision.reviewer,
        decision=decision.decision,
        review_seconds=decision.review_seconds,
        started_at=now,
        completed_at=now,
    )
    session.add(review_session)

    document.status = (
        DocumentStatus.REJECTED if decision.decision == "rejected" else DocumentStatus.REVIEWED
    )
    session.commit()
    return {"status": "recorded", "document_status": document.status.value}


class BulkReviewRequest(BaseModel):
    document_ids: list[str]
    reviewer: str
    decision: str
    review_seconds: float = 0.0


@queue_router.post("/bulk")
def bulk_review(request: BulkReviewRequest, session: Session = Depends(get_session)) -> dict:
    updated = []
    now = datetime.utcnow()
    for document_id in request.document_ids:
        document = session.get(Document, document_id)
        if document is None:
            continue
        session.add(
            ReviewSession(
                document_id=document_id,
                reviewer=request.reviewer,
                decision=request.decision,
                review_seconds=request.review_seconds,
                started_at=now,
                completed_at=now,
            )
        )
        document.status = (
            DocumentStatus.REJECTED if request.decision == "rejected" else DocumentStatus.REVIEWED
        )
        updated.append(document_id)

    session.commit()
    return {"updated": updated}
