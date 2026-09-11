from app.extraction.schemas import DOCUMENT_SCHEMAS, FEW_SHOT_EXAMPLES, Invoice


def test_invoice_schema_accepts_partial_data():
    invoice = Invoice(vendor_name="Acme", total_amount=100.0)
    assert invoice.vendor_name == "Acme"
    assert invoice.invoice_number is None
    assert invoice.line_items == []


def test_all_registered_schemas_have_few_shot_examples():
    for document_type in DOCUMENT_SCHEMAS:
        assert document_type in FEW_SHOT_EXAMPLES
        assert len(FEW_SHOT_EXAMPLES[document_type]) >= 1


def test_few_shot_examples_validate_against_their_schema():
    for document_type, schema in DOCUMENT_SCHEMAS.items():
        for example in FEW_SHOT_EXAMPLES[document_type]:
            # Should not raise -- every example must be schema-valid.
            schema(**example["output"])
