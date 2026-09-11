from datetime import date, timedelta

from app.extraction.schemas import Contract, Invoice, TerminationClause
from app.validation.business_rules import validate_business_rules


def test_unknown_vendor_flagged_as_warning():
    invoice = Invoice(vendor_name="Totally Unknown Co", total_amount=100.0)
    results = validate_business_rules("invoice", invoice)
    assert any(r.rule == "known_vendor" for r in results)


def test_known_vendor_out_of_range_total_flagged():
    invoice = Invoice(vendor_name="Acme Supplies Inc.", total_amount=999999.0)
    results = validate_business_rules("invoice", invoice)
    assert any(r.rule == "total_amount_range" for r in results)


def test_known_vendor_in_range_total_not_flagged():
    invoice = Invoice(vendor_name="Acme Supplies Inc.", total_amount=200.0, payment_terms="Net 30")
    results = validate_business_rules("invoice", invoice)
    assert not any(r.rule == "total_amount_range" for r in results)


def test_contract_missing_required_clauses_is_failure():
    contract = Contract(parties=["A", "B"], effective_date=date.today() + timedelta(days=10))
    results = validate_business_rules("contract", contract)
    assert any(r.rule == "required_clauses" and r.severity == "failure" for r in results)


def test_contract_broad_termination_clause_flagged():
    contract = Contract(
        parties=["A", "B"],
        effective_date=date.today() + timedelta(days=10),
        governing_law="Delaware",
        termination_clauses=[
            TerminationClause(text="Either party may terminate at any time for any reason.")
        ],
    )
    results = validate_business_rules("contract", contract)
    assert any(r.rule == "termination_clause_breadth" for r in results)
