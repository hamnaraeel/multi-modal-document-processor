"""Request/response models for the API (kept separate from the extraction
schemas in app.extraction.schemas, which describe document *content*)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    document_type: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    document_type: str | None
    page_image_urls: list[str]
    fields: dict | None
    field_confidence: dict | None
    field_sources: dict | None
    conflicts: list | None
    overall_confidence: float | None
    type_errors: list | None
    business_rule_warnings: list | None
    business_rule_failures: list | None
    anomalies: list | None
    routing_decision: str | None
    routing_reason: str | None


class CorrectionRequest(BaseModel):
    field_name: str
    original_value: str | None
    corrected_value: str | None
    reviewer: str
    correction_type: str  # "extraction_error" | "validation_false_positive" | "confirmed_correct"


class ReviewDecisionRequest(BaseModel):
    reviewer: str
    decision: str  # "approved" | "rejected" | "edited"
    review_seconds: float
