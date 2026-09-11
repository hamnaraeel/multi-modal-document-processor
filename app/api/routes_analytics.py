from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.analytics import metrics
from app.models.db import get_session

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/volume")
def volume(days: int = Query(30, le=365), session: Session = Depends(get_session)) -> list[dict]:
    return metrics.documents_processed_per_day(session, days)


@router.get("/auto-approval-rate")
def auto_approval_rate(days: int = Query(30, le=365), session: Session = Depends(get_session)) -> dict:
    return {
        "overall": metrics.overall_auto_approval_rate(session),
        "over_time": metrics.auto_approval_rate_over_time(session, days),
    }


@router.get("/extraction-accuracy")
def extraction_accuracy(session: Session = Depends(get_session)) -> list[dict]:
    return metrics.extraction_accuracy_by_field(session)


@router.get("/review-time")
def review_time(session: Session = Depends(get_session)) -> dict:
    return metrics.average_review_time(session)


@router.get("/ocr-engines")
def ocr_engines(session: Session = Depends(get_session)) -> dict:
    return metrics.ocr_engine_comparison(session)


@router.get("/confidence-distribution")
def confidence_distribution(session: Session = Depends(get_session)) -> list[dict]:
    return metrics.extraction_confidence_distribution(session)
