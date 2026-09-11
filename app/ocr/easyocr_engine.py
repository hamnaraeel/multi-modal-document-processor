from __future__ import annotations

import numpy as np
from PIL import Image

from app.ocr.tesseract_engine import OCRResult

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def run(image: Image.Image) -> OCRResult:
    reader = _get_reader()
    image_np = np.array(image.convert("RGB"))
    detections = reader.readtext(image_np)

    # Reading order: top-to-bottom, then left-to-right within a line band.
    def sort_key(detection):
        bbox, _text, _conf = detection
        top_left = bbox[0]
        return (round(top_left[1] / 15), top_left[0])

    detections.sort(key=sort_key)

    words = [text for _bbox, text, _conf in detections]
    confidences = [float(conf) for _bbox, _text, conf in detections]

    text = " ".join(words)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRResult(engine="easyocr", text=text, confidence=mean_confidence)
