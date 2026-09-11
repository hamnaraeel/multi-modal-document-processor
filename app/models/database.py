import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PREPROCESSING = "preprocessing"
    OCR_IN_PROGRESS = "ocr_in_progress"
    CLASSIFYING = "classifying"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    AUTO_APPROVED = "auto_approved"
    PENDING_FAST_REVIEW = "pending_fast_review"
    PENDING_DETAILED_REVIEW = "pending_detailed_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String)
    storage_path: Mapped[str] = mapped_column(String)
    mime_type: Mapped[str] = mapped_column(String)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    document_type: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), default=DocumentStatus.UPLOADED
    )
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    extraction: Mapped["ExtractionResult | None"] = relationship(
        back_populates="document", uselist=False, cascade="all, delete-orphan"
    )
    validation: Mapped["ValidationResult | None"] = relationship(
        back_populates="document", uselist=False, cascade="all, delete-orphan"
    )
    corrections: Mapped[list["Correction"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    page_number: Mapped[int] = mapped_column(Integer)
    image_path: Mapped[str | None] = mapped_column(String, nullable=True)

    # OCR
    native_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tesseract_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    easyocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tesseract_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    easyocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    agreement_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    merged_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    used_vision_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    extraction_strategy: Mapped[str | None] = mapped_column(String, nullable=True)

    # Preprocessing audit trail
    preprocessing_steps: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="pages")


class ExtractionResult(Base):
    __tablename__ = "extraction_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), unique=True)
    document_type: Mapped[str] = mapped_column(String)
    fields: Mapped[dict] = mapped_column(JSON)  # extracted field values
    field_confidence: Mapped[dict] = mapped_column(JSON)  # per-field confidence 0-1
    field_sources: Mapped[dict] = mapped_column(JSON)  # per-field source location(s)
    conflicts: Mapped[list] = mapped_column(JSON, default=list)  # chunk-merge conflicts
    overall_confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="extraction")


class ValidationResult(Base):
    __tablename__ = "validation_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), unique=True)
    type_errors: Mapped[list] = mapped_column(JSON, default=list)
    business_rule_warnings: Mapped[list] = mapped_column(JSON, default=list)
    business_rule_failures: Mapped[list] = mapped_column(JSON, default=list)
    anomalies: Mapped[list] = mapped_column(JSON, default=list)
    routing_decision: Mapped[str] = mapped_column(String)
    routing_reason: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="validation")


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    field_name: Mapped[str] = mapped_column(String)
    original_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str] = mapped_column(String)
    correction_type: Mapped[str] = mapped_column(
        String
    )  # "extraction_error" | "validation_false_positive" | "confirmed_correct"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="corrections")


class ReviewSession(Base):
    """Tracks a reviewer's decision on a document for throughput/accuracy analytics."""

    __tablename__ = "review_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    reviewer: Mapped[str] = mapped_column(String)
    decision: Mapped[str] = mapped_column(String)  # "approved" | "rejected" | "edited"
    review_seconds: Mapped[float] = mapped_column(Float)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    completed_at: Mapped[datetime] = mapped_column(DateTime)
