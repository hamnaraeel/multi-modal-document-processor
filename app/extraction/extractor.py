"""LLM extraction pipeline: classify -> chunk -> extract-per-chunk -> merge
-> score confidence -> locate source snippets.

The extraction prompt is built from the target Pydantic schema plus 2-3
few-shot examples for that document type, with an explicit instruction to
extract only what is present in the text (no inference, no defaulting).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from app.extraction.chunking import Chunk, FieldConflict, merge_chunk_extractions, split_into_chunks
from app.extraction.confidence import (
    compute_field_confidences,
    get_llm_self_reported_confidence,
    overall_confidence,
)
from app.extraction.schemas import DOCUMENT_SCHEMAS, FEW_SHOT_EXAMPLES
from app.llm.client import get_llm_client

NO_INFERENCE_INSTRUCTION = (
    "Extract ONLY information explicitly present in the text below. "
    "Do not infer, guess, or fill in defaults for missing fields -- leave "
    "them null/empty instead. Do not perform any calculations beyond "
    "reading values that are already written down."
)


@dataclass
class ExtractionOutput:
    document_type: str
    data: BaseModel
    field_confidence: dict[str, float]
    field_sources: dict[str, dict]
    conflicts: list[FieldConflict]
    overall_confidence: float
    chunks_used: int


def _build_messages(schema: type[BaseModel], document_type: str, chunk_text: str) -> list[dict]:
    examples = FEW_SHOT_EXAMPLES.get(document_type, [])
    example_blocks = []
    for ex in examples[:3]:
        example_blocks.append(
            f"Example input text:\n{ex['text']}\n\nExample expected output:\n{ex['output']}"
        )

    system_content = (
        f"You extract structured data from {document_type} documents into the given schema.\n"
        f"{NO_INFERENCE_INSTRUCTION}\n\n" + "\n\n---\n\n".join(example_blocks)
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Extract from this text:\n\n{chunk_text}"},
    ]


def extract_from_chunk(schema: type[BaseModel], document_type: str, chunk_text: str) -> BaseModel:
    client = get_llm_client()
    messages = _build_messages(schema, document_type, chunk_text)
    return client.structured_extract(schema, messages, max_tokens=4096)


def _locate_field_sources(schema_instance: BaseModel, chunks: list[Chunk]) -> dict[str, dict]:
    from app.extraction.confidence import _stringify  # local import to avoid cycle at module load

    sources: dict[str, dict] = {}
    for field_name in type(schema_instance).model_fields:
        value = getattr(schema_instance, field_name)
        value_str = _stringify(value).strip()
        if not value_str:
            sources[field_name] = {"snippet": None, "page_numbers": []}
            continue

        for chunk in chunks:
            if value_str.lower()[:40] in chunk.text.lower():
                sources[field_name] = {"snippet": value_str[:200], "page_numbers": chunk.page_numbers}
                break
        else:
            sources[field_name] = {"snippet": None, "page_numbers": []}

    return sources


def extract_document(document_type: str, pages: list[tuple[int, str]]) -> ExtractionOutput:
    schema = DOCUMENT_SCHEMAS[document_type]
    chunks = split_into_chunks(pages)

    per_chunk_results: list[tuple[int, BaseModel]] = [
        (chunk.index, extract_from_chunk(schema, document_type, chunk.text)) for chunk in chunks
    ]

    merged_instance, conflicts = merge_chunk_extractions(schema, per_chunk_results)

    full_text = "\n\n".join(chunk.text for chunk in chunks)
    llm_self_reported = get_llm_self_reported_confidence(schema, full_text)
    field_confidence = compute_field_confidences(merged_instance, full_text, conflicts, llm_self_reported)
    field_sources = _locate_field_sources(merged_instance, chunks)

    return ExtractionOutput(
        document_type=document_type,
        data=merged_instance,
        field_confidence=field_confidence,
        field_sources=field_sources,
        conflicts=conflicts,
        overall_confidence=overall_confidence(field_confidence),
        chunks_used=len(chunks),
    )
