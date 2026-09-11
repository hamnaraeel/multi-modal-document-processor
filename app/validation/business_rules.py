"""Domain-specific business rules, loaded from config/business_rules.yaml
so they can be tuned without a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

RULES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "business_rules.yaml"


@dataclass
class RuleResult:
    rule: str
    message: str
    severity: str  # "warning" | "failure"

    def as_dict(self) -> dict:
        return {"rule": self.rule, "message": self.message, "severity": self.severity}


def _load_rules() -> dict[str, Any]:
    with open(RULES_PATH) as f:
        return yaml.safe_load(f)


def _find_vendor(rules: dict, vendor_name: str | None) -> dict | None:
    if not vendor_name:
        return None
    for vendor in rules.get("known_vendors", []):
        if vendor["name"].strip().lower() == vendor_name.strip().lower():
            return vendor
    return None


def validate_invoice_business_rules(data: BaseModel) -> list[RuleResult]:
    rules = _load_rules()
    results: list[RuleResult] = []

    vendor_name = getattr(data, "vendor_name", None)
    vendor = _find_vendor(rules, vendor_name)

    if vendor_name and vendor is None:
        results.append(
            RuleResult(
                "known_vendor",
                f"Vendor '{vendor_name}' is not in the known vendor list",
                "warning",
            )
        )

    total_amount = getattr(data, "total_amount", None)
    if vendor and total_amount is not None:
        low, high = vendor["expected_total_range"]
        if not (low <= total_amount <= high):
            results.append(
                RuleResult(
                    "total_amount_range",
                    f"Total {total_amount} is outside {vendor_name}'s expected range [{low}, {high}]",
                    "warning",
                )
            )

    payment_terms = getattr(data, "payment_terms", None)
    if payment_terms:
        allowed = set(rules.get("standard_payment_terms_global", []))
        if vendor:
            allowed = set(vendor.get("standard_payment_terms", allowed))
        if payment_terms not in allowed:
            results.append(
                RuleResult(
                    "payment_terms",
                    f"Payment terms '{payment_terms}' are non-standard for this vendor",
                    "warning",
                )
            )

    return results


def validate_contract_business_rules(data: BaseModel) -> list[RuleResult]:
    rules = _load_rules()
    results: list[RuleResult] = []

    effective_date = getattr(data, "effective_date", None)
    if effective_date and effective_date < date.today():
        results.append(
            RuleResult(
                "effective_date",
                f"Effective date {effective_date.isoformat()} is in the past, not the future",
                "warning",
            )
        )

    required_clauses = set(rules.get("required_contract_clause_types", []))
    present: set[str] = set()
    if getattr(data, "termination_clauses", None):
        present.add("termination")
    if getattr(data, "governing_law", None):
        present.add("governing_law")
    if getattr(data, "key_obligations", None):
        present.add("obligations")

    for missing in required_clauses - present:
        results.append(
            RuleResult(
                "required_clauses",
                f"Contract is missing a required clause type: '{missing}'",
                "failure",
            )
        )

    broad_keywords = [kw.lower() for kw in rules.get("broad_termination_keywords", [])]
    for clause in getattr(data, "termination_clauses", None) or []:
        clause_text = (clause.text or "").lower()
        if any(kw in clause_text for kw in broad_keywords):
            results.append(
                RuleResult(
                    "termination_clause_breadth",
                    f"Termination clause reads as unusually broad: '{clause.text}'",
                    "warning",
                )
            )

    return results


def validate_business_rules(document_type: str, data: BaseModel) -> list[RuleResult]:
    if document_type in ("invoice", "receipt"):
        return validate_invoice_business_rules(data)
    if document_type == "contract":
        return validate_contract_business_rules(data)
    return []
