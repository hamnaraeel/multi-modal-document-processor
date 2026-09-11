"""Per-field extraction confidence scoring.

Each field's confidence blends four signals:
1. text_match   - does the extracted value literally (or fuzzily) appear
                   in the OCR text? Exact substring match scores 1.0;
                   partial/fuzzy matches score proportionally lower.
2. llm_reported - the LLM's own self-rated confidence for that field,
                   gathered via a follow-up structured call.
3. chunk_agreement - for multi-chunk documents, whether independent
                   chunks agreed on the value (1.0 if unanimous / only
                   one chunk saw it; degrades with more distinct values).
4. format_valid - whether the value satisfies basic type/format
                   expectations (dates parse, amounts are non-negative).

The final per-field score is the mean of whichever signals are available.
"""

from __future__ import annotations

import difflib
from datetime import date
from typing import Any

from loguru import logger
from pydantic import BaseModel, create_model

from app.extraction.chunking import FieldConflict
from app.llm.client import get_llm_client

NEGATIVE_INVALID_KEYWORDS = ("amount", "total", "price", "tax", "subtotal", "quantity")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, BaseModel):
        return " ".join(_stringify(v) for v in value.model_dump().values())
    if isinstance(value, list):
        return " ".join(_stringify(v) for v in value)
    return str(value)


def text_match_score(value: Any, ocr_text: str) -> float | None:
    value_str = _stringify(value).strip()
    if not value_str:
        return None
    ocr_lower = ocr_text.lower()
    if value_str.lower() in ocr_lower:
        return 1.0
    matcher = difflib.SequenceMatcher(None, value_str.lower(), ocr_lower, autojunk=False)
    match = matcher.find_longest_match(0, len(value_str), 0, len(ocr_lower))
    return round(match.size / len(value_str), 4) if value_str else 0.0


def format_validity_score(field_name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0
    if isinstance(value, (int, float)):
        if any(keyword in field_name.lower() for keyword in NEGATIVE_INVALID_KEYWORDS):
            return 1.0 if value >= 0 else 0.0
        return 1.0
    if isinstance(value, date):
        return 1.0 if date(1990, 1, 1) <= value <= date(2035, 12, 31) else 0.3
    return 1.0


def chunk_agreement_score(field_name: str, conflicts: list[FieldConflict]) -> float | None:
    for conflict in conflicts:
        if conflict.field == field_name:
            distinct = {v["value"] for v in conflict.values}
            return round(1.0 / max(len(distinct), 1), 4)
    return 1.0


def get_llm_self_reported_confidence(schema: type[BaseModel], ocr_text: str) -> dict[str, float]:
    """Ask the LLM to rate its own confidence per top-level field, 0-1."""
    field_names = list(schema.model_fields.keys())
    ConfidenceModel = create_model(
        "FieldConfidences",
        **{name: (float, 0.5) for name in field_names},  # type: ignore[arg-type]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "For each field below, rate 0-1 how explicitly and unambiguously that "
                "information appeared in the source text (1.0 = stated verbatim, "
                "0.0 = not present / had to be guessed)."
            ),
        },
        {
            "role": "user",
            "content": f"Fields: {field_names}\n\nSource text:\n{ocr_text[:6000]}",
        },
    ]

    try:
        client = get_llm_client()
        result = client.structured_extract(ConfidenceModel, messages, max_tokens=1024)
        return {name: float(getattr(result, name)) for name in field_names}
    except Exception as exc:  # LLM call is best-effort; never fail the pipeline on it
        logger.warning(f"LLM self-reported confidence unavailable: {exc}")
        return {}


def compute_field_confidences(
    schema_instance: BaseModel,
    ocr_text: str,
    conflicts: list[FieldConflict],
    llm_self_reported: dict[str, float] | None = None,
) -> dict[str, float]:
    llm_self_reported = llm_self_reported or {}
    confidences: dict[str, float] = {}

    for field_name in type(schema_instance).model_fields:
        value = getattr(schema_instance, field_name)
        signals = [
            score
            for score in (
                text_match_score(value, ocr_text),
                format_validity_score(field_name, value),
                chunk_agreement_score(field_name, conflicts),
                llm_self_reported.get(field_name),
            )
            if score is not None
        ]
        confidences[field_name] = round(sum(signals) / len(signals), 4) if signals else 0.0

    return confidences


def overall_confidence(field_confidences: dict[str, float]) -> float:
    if not field_confidences:
        return 0.0
    return round(sum(field_confidences.values()) / len(field_confidences), 4)
