from datetime import date

from app.extraction.schemas import Invoice, LineItem
from app.validation.type_validators import validate_type_level


def test_missing_required_field_flagged_as_error():
    invoice = Invoice(vendor_name=None, invoice_number="INV-1", total_amount=100.0)
    issues = validate_type_level("invoice", invoice)
    assert any(i.field == "vendor_name" and i.severity == "error" for i in issues)


def test_negative_amount_flagged_as_error():
    invoice = Invoice(vendor_name="Acme", invoice_number="INV-1", total_amount=-50.0)
    issues = validate_type_level("invoice", invoice)
    assert any(i.field == "total_amount" for i in issues)


def test_implausible_date_flagged():
    invoice = Invoice(
        vendor_name="Acme", invoice_number="INV-1", total_amount=100.0, invoice_date=date(1800, 1, 1)
    )
    issues = validate_type_level("invoice", invoice)
    assert any(i.field == "invoice_date" for i in issues)


def test_line_items_reconciliation_warning():
    invoice = Invoice(
        vendor_name="Acme",
        invoice_number="INV-1",
        total_amount=100.0,
        subtotal=100.0,
        line_items=[LineItem(description="A", quantity=1, unit_price=10, total=10)],
    )
    issues = validate_type_level("invoice", invoice)
    assert any(i.field == "line_items" and i.severity == "warning" for i in issues)


def test_valid_invoice_has_no_issues():
    invoice = Invoice(
        vendor_name="Acme",
        invoice_number="INV-1",
        total_amount=60.0,
        subtotal=60.0,
        line_items=[LineItem(description="A", quantity=1, unit_price=60, total=60)],
    )
    issues = validate_type_level("invoice", invoice)
    assert issues == []
