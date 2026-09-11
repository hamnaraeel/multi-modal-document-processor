"""Per-document-type extraction schemas.

These are intentionally lenient (most fields Optional) because the LLM
extractor must be able to return partial results for a document that is
missing some fields -- rejecting the whole extraction because one field
is absent would defeat the point of a confidence-based review pipeline.
Strict range/format/cross-field checks live in `app.validation`, run
*after* extraction, so a "missing field" and an "invalid field" produce
different, actionable signals.

To add a new document type: define its Pydantic model, add 2-3 few-shot
examples, and register both in DOCUMENT_SCHEMAS / FEW_SHOT_EXAMPLES below.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class SourcedValue(BaseModel):
    """Not used directly in LLM output, but documents the shape stored
    per-field in ExtractionResult.field_sources: where in the source text
    a value came from, for review-UI highlighting."""

    value: str
    source_snippet: str
    page_number: int | None = None


# ---------------------------------------------------------------------------
# Invoice
# ---------------------------------------------------------------------------
class LineItem(BaseModel):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    total: float | None = None


class Invoice(BaseModel):
    vendor_name: str | None = Field(None, description="Name of the vendor/seller issuing the invoice")
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total_amount: float | None = Field(None, description="Grand total amount due")
    payment_terms: str | None = Field(None, description="e.g. 'Net 30', 'Due on receipt'")
    currency: str | None = Field(None, description="ISO currency code if stated, e.g. USD")


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
class Obligation(BaseModel):
    party: str | None = None
    description: str | None = None


class TerminationClause(BaseModel):
    trigger: str | None = Field(None, description="Condition that allows termination")
    notice_period_days: int | None = None
    text: str | None = None


class Contract(BaseModel):
    parties: list[str] = Field(default_factory=list)
    effective_date: date | None = None
    term_length: str | None = Field(None, description="e.g. '12 months', 'until terminated'")
    key_obligations: list[Obligation] = Field(default_factory=list)
    termination_clauses: list[TerminationClause] = Field(default_factory=list)
    governing_law: str | None = None


# ---------------------------------------------------------------------------
# Receipt (lightweight third type, demonstrates schema extensibility)
# ---------------------------------------------------------------------------
class Receipt(BaseModel):
    merchant_name: str | None = None
    transaction_date: date | None = None
    items: list[LineItem] = Field(default_factory=list)
    total_amount: float | None = None
    payment_method: str | None = None


DocumentType = Literal["invoice", "contract", "receipt"]

DOCUMENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": Invoice,
    "contract": Contract,
    "receipt": Receipt,
}

# Two to three few-shot examples per document type, injected into the
# extraction prompt as (ocr_text_excerpt -> expected structured output).
FEW_SHOT_EXAMPLES: dict[str, list[dict]] = {
    "invoice": [
        {
            "text": (
                "Acme Supplies Inc.\nInvoice #INV-1042\nDate: 2024-03-01  "
                "Due: 2024-03-31\n\nDescription        Qty  Unit Price  Total\n"
                "Widget A            10       5.00      50.00\nWidget B             5      12.00      60.00\n\n"
                "Subtotal: 110.00\nTax: 8.80\nTotal: 118.80\nPayment Terms: Net 30"
            ),
            "output": {
                "vendor_name": "Acme Supplies Inc.",
                "invoice_number": "INV-1042",
                "invoice_date": "2024-03-01",
                "due_date": "2024-03-31",
                "line_items": [
                    {"description": "Widget A", "quantity": 10, "unit_price": 5.00, "total": 50.00},
                    {"description": "Widget B", "quantity": 5, "unit_price": 12.00, "total": 60.00},
                ],
                "subtotal": 110.00,
                "tax": 8.80,
                "total_amount": 118.80,
                "payment_terms": "Net 30",
                "currency": "USD",
            },
        },
        {
            "text": (
                "BILL TO: Jane Doe\nFROM: Northwind Traders\nInvoice No. 77-B\n"
                "Issued 12/05/2023\nTotal Due: $250.00 (due on receipt)"
            ),
            "output": {
                "vendor_name": "Northwind Traders",
                "invoice_number": "77-B",
                "invoice_date": "2023-12-05",
                "due_date": None,
                "line_items": [],
                "subtotal": None,
                "tax": None,
                "total_amount": 250.00,
                "payment_terms": "Due on receipt",
                "currency": "USD",
            },
        },
    ],
    "contract": [
        {
            "text": (
                "SERVICE AGREEMENT\nThis Agreement is made effective as of January 1, 2024 "
                "between Acme Corp ('Client') and Beta LLC ('Provider'). Term: 12 months. "
                "Provider shall deliver monthly reports. Either party may terminate with 30 "
                "days written notice. Governing law: State of Delaware."
            ),
            "output": {
                "parties": ["Acme Corp", "Beta LLC"],
                "effective_date": "2024-01-01",
                "term_length": "12 months",
                "key_obligations": [
                    {"party": "Beta LLC", "description": "Deliver monthly reports"}
                ],
                "termination_clauses": [
                    {
                        "trigger": "Either party may terminate",
                        "notice_period_days": 30,
                        "text": "Either party may terminate with 30 days written notice.",
                    }
                ],
                "governing_law": "State of Delaware",
            },
        }
    ],
    "receipt": [
        {
            "text": (
                "Corner Cafe\n05/02/2024\n1x Coffee  3.50\n1x Bagel   2.75\nTotal: 6.25\nPaid by card"
            ),
            "output": {
                "merchant_name": "Corner Cafe",
                "transaction_date": "2024-05-02",
                "items": [
                    {"description": "Coffee", "quantity": 1, "unit_price": 3.50, "total": 3.50},
                    {"description": "Bagel", "quantity": 1, "unit_price": 2.75, "total": 2.75},
                ],
                "total_amount": 6.25,
                "payment_method": "card",
            },
        }
    ],
}
