"""Type-level validation: required fields, valid ranges/formats, and
cross-field consistency. Returns a list of structured errors instead of
raising, so one bad field doesn't block the whole document from
reaching the review queue with the rest of its data intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel

REQUIRED_FIELDS: dict[str, list[str]] = {
    "invoice": ["vendor_name", "invoice_number", "total_amount"],
    "contract": ["parties", "effective_date"],
    "receipt": ["merchant_name", "total_amount"],
}

AMOUNT_TOLERANCE = 0.02  # 2% relative tolerance when reconciling sums


@dataclass
class ValidationIssue:
    field: str
    message: str
    severity: str  # "error" | "warning"

    def as_dict(self) -> dict:
        return {"field": self.field, "message": self.message, "severity": self.severity}


def check_required_fields(document_type: str, data: BaseModel) -> list[ValidationIssue]:
    issues = []
    for field_name in REQUIRED_FIELDS.get(document_type, []):
        value = getattr(data, field_name, None)
        is_missing = value is None or (isinstance(value, (list, str)) and len(value) == 0)
        if is_missing:
            issues.append(
                ValidationIssue(field_name, f"Required field '{field_name}' is missing", "error")
            )
    return issues


def check_date_plausibility(data: BaseModel) -> list[ValidationIssue]:
    issues = []
    today = date.today()
    for field_name in type(data).model_fields:
        value = getattr(data, field_name, None)
        if isinstance(value, date):
            if value.year < 1990 or value > date(today.year + 15, 12, 31):
                issues.append(
                    ValidationIssue(
                        field_name, f"Date {value.isoformat()} is outside a plausible range", "error"
                    )
                )
    return issues


def check_monetary_fields(data: BaseModel) -> list[ValidationIssue]:
    issues = []
    for field_name in ("total_amount", "subtotal", "tax"):
        value = getattr(data, field_name, None)
        if isinstance(value, (int, float)) and value < 0:
            issues.append(
                ValidationIssue(field_name, f"'{field_name}' cannot be negative ({value})", "error")
            )
    return issues


def check_line_items_sum_to_total(data: BaseModel) -> list[ValidationIssue]:
    """Invoice/Receipt-specific: line item totals should reconcile with the
    stated subtotal/total, within a small rounding tolerance."""
    line_items = getattr(data, "line_items", None) or getattr(data, "items", None)
    if not line_items:
        return []

    computed_sum = sum(item.total for item in line_items if item.total is not None)
    target = getattr(data, "subtotal", None) or getattr(data, "total_amount", None)
    if target is None or computed_sum == 0:
        return []

    if abs(computed_sum - target) > max(AMOUNT_TOLERANCE * target, 0.01):
        return [
            ValidationIssue(
                "line_items",
                f"Line items sum to {computed_sum:.2f} but subtotal/total is {target:.2f}",
                "warning",
            )
        ]
    return []


def validate_type_level(document_type: str, data: BaseModel) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues += check_required_fields(document_type, data)
    issues += check_date_plausibility(data)
    issues += check_monetary_fields(data)
    if document_type in ("invoice", "receipt"):
        issues += check_line_items_sum_to_total(data)
    return issues
