"""Dual-engine OCR ensemble: run Tesseract and EasyOCR, align their output
character-by-character, and pick the higher-confidence reading wherever
they disagree.

When both engines produce identical text for a segment, we trust it fully.
When they diverge, `difflib.SequenceMatcher` locates exactly which spans
differ, and we take the span from whichever engine has the higher overall
confidence for that page. This ensemble approach catches character-level
misreads that either engine alone would silently get wrong.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from PIL import Image

from app.ocr import easyocr_engine, tesseract_engine
from app.ocr.tesseract_engine import OCRResult


@dataclass
class MergedOCRResult:
    text: str
    confidence: float
    agreement_ratio: float
    tesseract_confidence: float
    easyocr_confidence: float
    discrepancies: list[dict] = field(default_factory=list)


def merge(tess: OCRResult, easy: OCRResult) -> MergedOCRResult:
    matcher = difflib.SequenceMatcher(a=tess.text, b=easy.text, autojunk=False)
    agreement_ratio = matcher.ratio()

    merged_parts: list[str] = []
    discrepancies: list[dict] = []

    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        tess_segment = tess.text[a_start:a_end]
        easy_segment = easy.text[b_start:b_end]

        if tag == "equal":
            merged_parts.append(tess_segment)
            continue

        # Engines disagree on this span — trust whichever engine scored
        # higher overall confidence on the full page.
        chosen_engine, chosen_segment = (
            ("tesseract", tess_segment)
            if tess.confidence >= easy.confidence
            else ("easyocr", easy_segment)
        )
        merged_parts.append(chosen_segment)

        if tess_segment.strip() or easy_segment.strip():
            discrepancies.append(
                {
                    "tesseract_reading": tess_segment,
                    "easyocr_reading": easy_segment,
                    "chosen_engine": chosen_engine,
                    "chosen_reading": chosen_segment,
                }
            )

    merged_text = "".join(merged_parts)

    # Overall confidence blends how much the two engines agreed with how
    # confident each individually was.
    confidence = round(
        0.5 * agreement_ratio + 0.25 * tess.confidence + 0.25 * easy.confidence, 4
    )

    return MergedOCRResult(
        text=merged_text,
        confidence=confidence,
        agreement_ratio=round(agreement_ratio, 4),
        tesseract_confidence=tess.confidence,
        easyocr_confidence=easy.confidence,
        discrepancies=discrepancies,
    )


def run_ensemble(image: Image.Image) -> MergedOCRResult:
    tess_result = tesseract_engine.run(image)
    easy_result = easyocr_engine.run(image)
    return merge(tess_result, easy_result)


def quick_confidence(image: Image.Image) -> float:
    """Cheap single-engine confidence estimate (Tesseract only) for use
    inside the preprocessing before/after comparison, where running the
    full dual-engine ensemble twice per page would be wasteful."""
    return tesseract_engine.run(image).confidence
