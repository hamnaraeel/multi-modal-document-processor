from datetime import date

from app.extraction.confidence import format_validity_score, text_match_score


def test_exact_text_match_scores_one():
    assert text_match_score("Acme Supplies Inc.", "Vendor: Acme Supplies Inc. Total: 100") == 1.0


def test_missing_text_scores_low():
    score = text_match_score("Totally Different Vendor", "Vendor: Acme Supplies Inc.")
    assert score < 0.5


def test_none_value_returns_none():
    assert text_match_score(None, "some text") is None


def test_negative_amount_field_invalid():
    assert format_validity_score("total_amount", -5.0) == 0.0


def test_positive_amount_field_valid():
    assert format_validity_score("total_amount", 5.0) == 1.0


def test_implausible_date_scores_low():
    assert format_validity_score("invoice_date", date(1800, 1, 1)) < 1.0


def test_plausible_date_scores_high():
    assert format_validity_score("invoice_date", date(2024, 1, 1)) == 1.0
