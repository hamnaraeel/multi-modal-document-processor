from __future__ import annotations

from pydantic import BaseModel, Field

from app.extraction.schemas import DOCUMENT_SCHEMAS
from app.llm.client import get_llm_client


class ClassificationResult(BaseModel):
    document_type: str = Field(description="One of the known document type labels")
    confidence: float = Field(ge=0, le=1, description="Self-reported classification confidence")
    reasoning: str = Field(description="One-sentence justification")


def classify_document(ocr_text: str) -> ClassificationResult:
    known_types = list(DOCUMENT_SCHEMAS.keys())
    excerpt = ocr_text[:4000]

    messages = [
        {
            "role": "system",
            "content": (
                "You classify business documents from their OCR text into exactly one "
                f"of these types: {known_types}. If none fit well, still pick the closest "
                "one and lower your confidence accordingly."
            ),
        },
        {"role": "user", "content": f"Document text:\n\n{excerpt}"},
    ]

    client = get_llm_client()
    result = client.structured_extract(ClassificationResult, messages, max_tokens=300)

    if result.document_type not in known_types:
        # Guard against the model inventing a label outside the known set.
        result.document_type = known_types[0]
        result.confidence = min(result.confidence, 0.3)

    return result
