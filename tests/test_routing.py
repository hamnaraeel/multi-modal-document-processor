from app.models.database import DocumentStatus
from app.validation.routing import route_document
from app.validation.type_validators import ValidationIssue


def test_high_confidence_no_issues_auto_approves():
    decision = route_document(0.95, [], [], [])
    assert decision.status == DocumentStatus.AUTO_APPROVED


def test_medium_confidence_routes_to_fast_review():
    decision = route_document(0.80, [], [], [])
    assert decision.status == DocumentStatus.PENDING_FAST_REVIEW


def test_low_confidence_routes_to_detailed_review():
    decision = route_document(0.40, [], [], [])
    assert decision.status == DocumentStatus.PENDING_DETAILED_REVIEW


def test_critical_error_forces_detailed_review_even_with_high_confidence():
    issues = [ValidationIssue(field="total_amount", message="negative", severity="error")]
    decision = route_document(0.99, issues, [], [])
    assert decision.status == DocumentStatus.PENDING_DETAILED_REVIEW
