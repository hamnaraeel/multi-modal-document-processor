"""Statistical anomaly detection against historical extractions for the
same document type + source. Flags amounts that are far outside the
typical range, and notes when there's no history yet to compare against.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.database import Document, ExtractionResult

Z_SCORE_THRESHOLD = 2.5
MIN_HISTORY_FOR_STATS = 3


@dataclass
class AnomalyResult:
    kind: str
    message: str
    severity: str  # "info" | "warning"

    def as_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message, "severity": self.severity}


def _historical_totals(
    session: Session, document_type: str, source: str | None, exclude_document_id: str | None
) -> list[float]:
    query = (
        select(ExtractionResult.fields)
        .join(Document, Document.id == ExtractionResult.document_id)
        .where(Document.document_type == document_type)
    )
    if source:
        query = query.where(Document.source == source)
    if exclude_document_id:
        query = query.where(Document.id != exclude_document_id)

    totals = []
    for (fields,) in session.execute(query).all():
        amount = (fields or {}).get("total_amount")
        if isinstance(amount, (int, float)):
            totals.append(float(amount))
    return totals


def detect_amount_anomaly(
    session: Session,
    document_type: str,
    source: str | None,
    data: BaseModel,
    exclude_document_id: str | None = None,
) -> list[AnomalyResult]:
    current_amount = getattr(data, "total_amount", None)
    if current_amount is None:
        return []

    history = _historical_totals(session, document_type, source, exclude_document_id)

    if len(history) < MIN_HISTORY_FOR_STATS:
        return [
            AnomalyResult(
                "insufficient_history",
                f"Only {len(history)} prior {document_type}(s) from this source -- "
                "not enough history yet for statistical comparison",
                "info",
            )
        ]

    mean = statistics.mean(history)
    stdev = statistics.stdev(history)
    if stdev == 0:
        return []

    z_score = (current_amount - mean) / stdev
    if abs(z_score) > Z_SCORE_THRESHOLD:
        direction = "higher" if z_score > 0 else "lower"
        return [
            AnomalyResult(
                "amount_outlier",
                f"Total amount {current_amount:.2f} is unusually {direction} than this "
                f"source's historical average of {mean:.2f} (z-score {z_score:.2f})",
                "warning",
            )
        ]
    return []


def detect_new_vendor(
    session: Session, document_type: str, vendor_name: str | None
) -> list[AnomalyResult]:
    if not vendor_name:
        return []

    # A vendor-name existence check inside the fields JSON column is
    # DB-dependent, so we filter in Python over a bounded scan instead.
    rows = session.execute(
        select(ExtractionResult.fields).join(Document, Document.id == ExtractionResult.document_id)
        .where(Document.document_type == document_type)
    ).all()
    seen_before = any((fields or {}).get("vendor_name") == vendor_name for (fields,) in rows)

    if not seen_before:
        return [
            AnomalyResult(
                "new_vendor",
                f"'{vendor_name}' has not appeared in any prior processed document",
                "info",
            )
        ]
    return []


def detect_anomalies(
    session: Session,
    document_type: str,
    source: str | None,
    data: BaseModel,
    exclude_document_id: str | None = None,
) -> list[AnomalyResult]:
    results: list[AnomalyResult] = []
    results += detect_amount_anomaly(session, document_type, source, data, exclude_document_id)
    if document_type in ("invoice", "receipt"):
        vendor_name = getattr(data, "vendor_name", None) or getattr(data, "merchant_name", None)
        results += detect_new_vendor(session, document_type, vendor_name)
    return results
