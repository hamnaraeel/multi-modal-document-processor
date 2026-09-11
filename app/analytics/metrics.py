"""Operational analytics: the metrics that tell the story of the
pipeline's maturity -- volume, auto-approval rate, extraction accuracy,
review throughput, and OCR engine performance.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.database import (
    Correction,
    Document,
    DocumentPage,
    DocumentStatus,
    ExtractionResult,
    ReviewSession,
)

AUTO_APPROVED_LIKE = {DocumentStatus.AUTO_APPROVED}
TERMINAL_STATUSES = {
    DocumentStatus.AUTO_APPROVED,
    DocumentStatus.REVIEWED,
    DocumentStatus.REJECTED,
}


def documents_processed_per_day(session: Session, days: int = 30) -> list[dict]:
    since = datetime.utcnow() - timedelta(days=days)
    rows = session.execute(
        select(Document.created_at, Document.status).where(Document.created_at >= since)
    ).all()

    counts: dict[str, int] = defaultdict(int)
    for created_at, _status in rows:
        counts[created_at.date().isoformat()] += 1

    return [{"date": d, "count": c} for d, c in sorted(counts.items())]


def auto_approval_rate_over_time(session: Session, days: int = 30) -> list[dict]:
    since = datetime.utcnow() - timedelta(days=days)
    rows = session.execute(
        select(Document.created_at, Document.status)
        .where(Document.created_at >= since)
        .where(Document.status.in_(TERMINAL_STATUSES))
    ).all()

    by_day_total: dict[str, int] = defaultdict(int)
    by_day_auto: dict[str, int] = defaultdict(int)
    for created_at, status in rows:
        day = created_at.date().isoformat()
        by_day_total[day] += 1
        if status in AUTO_APPROVED_LIKE:
            by_day_auto[day] += 1

    return [
        {"date": day, "auto_approval_rate": round(by_day_auto[day] / total, 4) if total else 0.0}
        for day, total in sorted(by_day_total.items())
    ]


def overall_auto_approval_rate(session: Session) -> float:
    total = session.execute(
        select(func.count()).select_from(Document).where(Document.status.in_(TERMINAL_STATUSES))
    ).scalar_one()
    if not total:
        return 0.0
    auto = session.execute(
        select(func.count()).select_from(Document).where(Document.status == DocumentStatus.AUTO_APPROVED)
    ).scalar_one()
    return round(auto / total, 4)


def extraction_accuracy_by_field(session: Session) -> list[dict]:
    """Accuracy proxy from human corrections: a field marked
    'extraction_error' means the original extraction was wrong; a field
    marked 'confirmed_correct' or 'validation_false_positive' means the
    extraction itself was right."""
    rows = session.execute(
        select(Correction.field_name, Correction.correction_type, Document.document_type)
        .join(Document, Document.id == Correction.document_id)
    ).all()

    stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0})
    for field_name, correction_type, document_type in rows:
        key = (document_type or "unknown", field_name)
        if correction_type == "extraction_error":
            stats[key]["wrong"] += 1
        else:
            stats[key]["correct"] += 1

    results = []
    for (document_type, field_name), counts in stats.items():
        total = counts["correct"] + counts["wrong"]
        results.append(
            {
                "document_type": document_type,
                "field": field_name,
                "accuracy": round(counts["correct"] / total, 4) if total else None,
                "sample_size": total,
            }
        )
    return sorted(results, key=lambda r: (r["document_type"], r["field"]))


def average_review_time(session: Session) -> dict:
    rows = session.execute(
        select(ReviewSession.review_seconds, Document.document_type)
        .join(Document, Document.id == ReviewSession.document_id)
    ).all()

    if not rows:
        return {"overall_avg_seconds": None, "by_document_type": []}

    overall_avg = sum(r[0] for r in rows) / len(rows)

    by_type: dict[str, list[float]] = defaultdict(list)
    for review_seconds, document_type in rows:
        by_type[document_type or "unknown"].append(review_seconds)

    return {
        "overall_avg_seconds": round(overall_avg, 2),
        "by_document_type": [
            {"document_type": t, "avg_seconds": round(sum(v) / len(v), 2), "sample_size": len(v)}
            for t, v in by_type.items()
        ],
    }


def ocr_engine_comparison(session: Session) -> dict:
    rows = session.execute(
        select(
            DocumentPage.tesseract_confidence,
            DocumentPage.easyocr_confidence,
            DocumentPage.agreement_ratio,
            DocumentPage.used_vision_fallback,
        ).where(DocumentPage.extraction_strategy == "ocr_ensemble")
    ).all()

    fallback_rows = session.execute(select(DocumentPage.used_vision_fallback)).all()

    if not rows:
        tess_avg = easy_avg = agreement_avg = None
    else:
        tess_vals = [r[0] for r in rows if r[0] is not None]
        easy_vals = [r[1] for r in rows if r[1] is not None]
        agreement_vals = [r[2] for r in rows if r[2] is not None]
        tess_avg = round(sum(tess_vals) / len(tess_vals), 4) if tess_vals else None
        easy_avg = round(sum(easy_vals) / len(easy_vals), 4) if easy_vals else None
        agreement_avg = round(sum(agreement_vals) / len(agreement_vals), 4) if agreement_vals else None

    total_pages = len(fallback_rows)
    fallback_pages = sum(1 for (used,) in fallback_rows if used)

    return {
        "avg_tesseract_confidence": tess_avg,
        "avg_easyocr_confidence": easy_avg,
        "avg_engine_agreement_ratio": agreement_avg,
        "vision_fallback_rate": round(fallback_pages / total_pages, 4) if total_pages else 0.0,
        "pages_analyzed": total_pages,
    }


def extraction_confidence_distribution(session: Session) -> list[dict]:
    rows = session.execute(select(ExtractionResult.document_type, ExtractionResult.overall_confidence)).all()
    return [{"document_type": t, "overall_confidence": c} for t, c in rows]
