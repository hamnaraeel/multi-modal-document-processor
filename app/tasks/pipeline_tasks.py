"""The end-to-end document processing pipeline, run as a single Celery task
per document. Each stage updates the document's status so the API/review
UI can poll progress, and every OCR/preprocessing/extraction decision is
persisted for audit and analytics.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from app.config import settings
from app.extraction import classifier, extractor
from app.ingestion import loader, preprocessing
from app.models.database import (
    Document,
    DocumentPage,
    DocumentStatus,
    ExtractionResult,
    ValidationResult,
)
from app.models.db import SessionLocal
from app.ocr import easyocr_engine, ensemble, tesseract_engine, vision_fallback
from app.tasks.celery_app import celery_app
from app.validation import anomaly_detector, business_rules, routing, type_validators


def _save_page_image(document_id: str, page_number: int, image) -> str:
    out_dir = Path(settings.processed_dir) / document_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{page_number}.png"
    image.save(out_path, format="PNG")
    return str(out_path)


def _process_page(document_id: str, page) -> DocumentPage:
    db_page = DocumentPage(document_id=document_id, page_number=page.page_number)
    db_page.image_path = _save_page_image(document_id, page.page_number, page.image)

    if not page.needs_ocr:
        db_page.native_text = page.native_text
        db_page.merged_text = page.native_text
        db_page.ocr_confidence = 1.0
        db_page.extraction_strategy = "native_pdf_text"
        db_page.preprocessing_steps = {"steps_applied": [], "reason": "native text layer present"}
        return db_page

    processed_image, report = preprocessing.preprocess_with_confidence_tracking(
        page.image, ensemble.quick_confidence
    )
    db_page.preprocessing_steps = report.as_dict()

    tess_result = tesseract_engine.run(processed_image)
    easy_result = easyocr_engine.run(processed_image)
    merged = ensemble.merge(tess_result, easy_result)

    db_page.tesseract_text = tess_result.text
    db_page.easyocr_text = easy_result.text
    db_page.tesseract_confidence = tess_result.confidence
    db_page.easyocr_confidence = easy_result.confidence
    db_page.agreement_ratio = merged.agreement_ratio
    db_page.merged_text = merged.text
    db_page.ocr_confidence = merged.confidence
    db_page.extraction_strategy = "ocr_ensemble"

    if vision_fallback.should_use_vision_fallback(merged.confidence, merged.agreement_ratio):
        try:
            vision_text = vision_fallback.extract_text_with_vision(page.image)
            if vision_text.strip():
                db_page.merged_text = vision_text
                db_page.used_vision_fallback = True
                db_page.extraction_strategy = "vision_fallback"
                db_page.ocr_confidence = max(merged.confidence, 0.75)
        except Exception as exc:
            logger.warning(f"Vision fallback failed for page {page.page_number}: {exc}")

    return db_page


@celery_app.task(name="app.tasks.process_document", bind=True)
def process_document(self, document_id: str) -> dict:
    session = SessionLocal()
    try:
        document = session.get(Document, document_id)
        if document is None:
            raise ValueError(f"Document {document_id} not found")

        # --- Phase 1: ingestion + OCR ---
        document.status = DocumentStatus.PREPROCESSING
        session.commit()

        loaded = loader.load_document(document.storage_path)
        document.page_count = loaded.page_count
        document.status = DocumentStatus.OCR_IN_PROGRESS
        session.commit()

        db_pages = [_process_page(document_id, page) for page in loaded.pages]
        session.add_all(db_pages)
        session.commit()

        full_pages_text = [(p.page_number, p.merged_text or "") for p in db_pages]
        full_text = "\n\n".join(text for _n, text in full_pages_text)

        # --- Phase 2: classification + LLM extraction ---
        document.status = DocumentStatus.CLASSIFYING
        session.commit()

        classification = classifier.classify_document(full_text)
        document.document_type = classification.document_type

        document.status = DocumentStatus.EXTRACTING
        session.commit()

        extraction = extractor.extract_document(classification.document_type, full_pages_text)

        db_extraction = ExtractionResult(
            document_id=document_id,
            document_type=extraction.document_type,
            fields=extraction.data.model_dump(mode="json"),
            field_confidence=extraction.field_confidence,
            field_sources=extraction.field_sources,
            conflicts=[c.__dict__ for c in extraction.conflicts],
            overall_confidence=extraction.overall_confidence,
        )
        session.add(db_extraction)
        session.commit()

        # --- Phase 3: validation + routing ---
        document.status = DocumentStatus.VALIDATING
        session.commit()

        type_issues = type_validators.validate_type_level(extraction.document_type, extraction.data)
        rule_results = business_rules.validate_business_rules(extraction.document_type, extraction.data)
        anomalies = anomaly_detector.detect_anomalies(
            session,
            extraction.document_type,
            document.source,
            extraction.data,
            exclude_document_id=document_id,
        )

        decision = routing.route_document(
            extraction.overall_confidence, type_issues, rule_results, anomalies
        )

        db_validation = ValidationResult(
            document_id=document_id,
            type_errors=[i.as_dict() for i in type_issues],
            business_rule_warnings=[r.as_dict() for r in rule_results if r.severity == "warning"],
            business_rule_failures=[r.as_dict() for r in rule_results if r.severity == "failure"],
            anomalies=[a.as_dict() for a in anomalies],
            routing_decision=decision.status.value,
            routing_reason=decision.reason,
        )
        session.add(db_validation)

        document.status = decision.status
        session.commit()

        return {
            "document_id": document_id,
            "status": document.status.value,
            "document_type": extraction.document_type,
            "overall_confidence": extraction.overall_confidence,
        }

    except Exception as exc:
        logger.exception(f"Pipeline failed for document {document_id}")
        session.rollback()
        document = session.get(Document, document_id)
        if document is not None:
            document.status = DocumentStatus.FAILED
            document.error_message = str(exc)
            session.commit()
        raise
    finally:
        session.close()
