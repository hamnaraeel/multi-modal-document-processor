"""Vision-model fallback for pages where traditional OCR struggles:
handwriting, poor scan quality, or complex layouts like tables.

This is more expensive per page than Tesseract/EasyOCR, so it should only
be invoked when the ensemble's confidence or agreement ratio is low.
"""

from __future__ import annotations

import io

from PIL import Image

from app.llm.client import get_llm_client

VISION_OCR_PROMPT = """You are an OCR engine. Extract ALL text from this document \
page image exactly as it appears, preserving the original structure as closely \
as possible:

- Preserve tables as pipe-delimited rows (| col1 | col2 | ...).
- Preserve line breaks and paragraph structure.
- Include headers, footers, and any handwritten text you can read.
- Do not summarize, translate, or omit anything.
- Do not add commentary before or after the extracted text.

Output only the extracted text."""

CONFIDENCE_TRIGGER_THRESHOLD = 0.55
AGREEMENT_TRIGGER_THRESHOLD = 0.5


def should_use_vision_fallback(ensemble_confidence: float, agreement_ratio: float) -> bool:
    return (
        ensemble_confidence < CONFIDENCE_TRIGGER_THRESHOLD
        or agreement_ratio < AGREEMENT_TRIGGER_THRESHOLD
    )


def extract_text_with_vision(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    client = get_llm_client()
    return client.vision_extract_text(image_bytes, VISION_OCR_PROMPT)
