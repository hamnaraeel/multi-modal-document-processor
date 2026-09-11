from app.ocr.ensemble import merge
from app.ocr.tesseract_engine import OCRResult


def test_identical_engine_outputs_yield_high_agreement():
    tess = OCRResult(engine="tesseract", text="Invoice Total: 118.80", confidence=0.9)
    easy = OCRResult(engine="easyocr", text="Invoice Total: 118.80", confidence=0.88)
    result = merge(tess, easy)
    assert result.agreement_ratio == 1.0
    assert result.text == "Invoice Total: 118.80"
    assert result.discrepancies == []


def test_disagreement_picks_higher_confidence_engine_segment():
    tess = OCRResult(engine="tesseract", text="Total: 118.80", confidence=0.95)
    easy = OCRResult(engine="easyocr", text="Total: 11B.80", confidence=0.60)
    result = merge(tess, easy)
    assert "118.80" in result.text
    assert len(result.discrepancies) >= 1
    assert result.discrepancies[0]["chosen_engine"] == "tesseract"


def test_low_agreement_lowers_overall_confidence():
    tess = OCRResult(engine="tesseract", text="abc def ghi", confidence=0.9)
    easy = OCRResult(engine="easyocr", text="xyz uvw rst", confidence=0.9)
    result = merge(tess, easy)
    assert result.agreement_ratio < 0.3
    assert result.confidence < 0.9
