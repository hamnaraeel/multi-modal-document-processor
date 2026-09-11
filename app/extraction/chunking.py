"""Chunk-and-merge for documents longer than the LLM's comfortable context.

Long documents are split by page into chunks under a character budget.
Each chunk is extracted independently (see extractor.py), and this module
merges the per-chunk Pydantic results back into one:

- List fields (line items, parties, obligations, ...) are concatenated
  and de-duplicated.
- Scalar fields (vendor_name, total_amount, ...) are expected to agree
  across chunks that both saw a value. If they don't, that's a genuine
  conflict -- both values are kept, tagged with their source chunk, and
  surfaced to the reviewer rather than silently picking one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, get_origin

from pydantic import BaseModel

CHUNK_CHAR_BUDGET = 12000


@dataclass
class Chunk:
    index: int
    page_numbers: list[int]
    text: str


@dataclass
class FieldConflict:
    field: str
    values: list[dict] = field(default_factory=list)


def split_into_chunks(pages: list[tuple[int, str]], max_chars: int = CHUNK_CHAR_BUDGET) -> list[Chunk]:
    """`pages` is a list of (page_number, page_text)."""
    chunks: list[Chunk] = []
    current_pages: list[int] = []
    current_text_parts: list[str] = []
    current_len = 0

    def flush():
        if current_text_parts:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    page_numbers=list(current_pages),
                    text="\n\n".join(current_text_parts),
                )
            )

    for page_number, page_text in pages:
        page_text = page_text or ""
        if current_len + len(page_text) > max_chars and current_text_parts:
            flush()
            current_pages, current_text_parts, current_len = [], [], 0

        current_pages.append(page_number)
        current_text_parts.append(page_text)
        current_len += len(page_text)

    flush()
    return chunks or [Chunk(index=0, page_numbers=[], text="")]


def _is_list_field(schema: type[BaseModel], field_name: str) -> bool:
    annotation = schema.model_fields[field_name].annotation
    return get_origin(annotation) is list


def _dedupe_key(item: Any) -> str:
    if isinstance(item, BaseModel):
        return item.model_dump_json()
    return str(item)


def merge_chunk_extractions(
    schema: type[BaseModel],
    per_chunk_results: list[tuple[int, BaseModel]],
) -> tuple[BaseModel, list[FieldConflict]]:
    if len(per_chunk_results) == 1:
        return per_chunk_results[0][1], []

    merged: dict[str, Any] = {}
    conflicts: list[FieldConflict] = []

    for field_name in schema.model_fields:
        values_by_chunk = [(idx, getattr(result, field_name)) for idx, result in per_chunk_results]

        if _is_list_field(schema, field_name):
            merged_list: list[Any] = []
            seen: set[str] = set()
            for _idx, value in values_by_chunk:
                for item in value or []:
                    key = _dedupe_key(item)
                    if key not in seen:
                        seen.add(key)
                        merged_list.append(item)
            merged[field_name] = merged_list
            continue

        non_null = [(idx, v) for idx, v in values_by_chunk if v is not None]
        if not non_null:
            merged[field_name] = None
            continue

        distinct = {_dedupe_key(v) for _idx, v in non_null}
        merged[field_name] = non_null[0][1]

        if len(distinct) > 1:
            conflicts.append(
                FieldConflict(
                    field=field_name,
                    values=[
                        {"chunk_index": idx, "value": v.isoformat() if hasattr(v, "isoformat") else v}
                        for idx, v in non_null
                    ],
                )
            )

    return schema(**merged), conflicts
