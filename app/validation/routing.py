"""Confidence-based routing: decide whether a document auto-approves or
goes to a human review queue, and if so, which one.

- High confidence + all validations pass         -> auto-approve
- Medium confidence, or only warnings             -> fast review queue
- Low confidence, or any critical (error/failure) -> detailed review queue
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.models.database import DocumentStatus
from app.validation.anomaly_detector import AnomalyResult
from app.validation.business_rules import RuleResult
from app.validation.type_validators import ValidationIssue


@dataclass
class RoutingDecision:
    status: DocumentStatus
    reason: str


def route_document(
    overall_confidence: float,
    type_issues: list[ValidationIssue],
    rule_results: list[RuleResult],
    anomalies: list[AnomalyResult],
) -> RoutingDecision:
    has_critical_failure = any(i.severity == "error" for i in type_issues) or any(
        r.severity == "failure" for r in rule_results
    )
    has_warning = (
        any(i.severity == "warning" for i in type_issues)
        or any(r.severity == "warning" for r in rule_results)
        or any(a.severity == "warning" for a in anomalies)
    )

    if has_critical_failure:
        return RoutingDecision(
            DocumentStatus.PENDING_DETAILED_REVIEW,
            "Critical validation failure(s) present -- requires detailed human review",
        )

    if overall_confidence < settings.fast_review_confidence_threshold:
        return RoutingDecision(
            DocumentStatus.PENDING_DETAILED_REVIEW,
            f"Overall confidence {overall_confidence:.2f} is below the detailed-review "
            f"threshold of {settings.fast_review_confidence_threshold:.2f}",
        )

    if overall_confidence < settings.auto_approve_confidence_threshold or has_warning:
        return RoutingDecision(
            DocumentStatus.PENDING_FAST_REVIEW,
            f"Overall confidence {overall_confidence:.2f} or minor validation warnings "
            "require a quick human check",
        )

    return RoutingDecision(
        DocumentStatus.AUTO_APPROVED,
        f"Overall confidence {overall_confidence:.2f} with no validation issues",
    )
