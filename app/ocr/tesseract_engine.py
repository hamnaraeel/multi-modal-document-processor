from __future__ import annotations

from dataclasses import dataclass

import pytesseract
from PIL import Image


@dataclass
class OCRResult:
    engine: str
    text: str
    confidence: float  # normalized 0-1


def run(image: Image.Image) -> OCRResult:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words: list[str] = []
    confidences: list[float] = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        conf = float(conf)
        if word.strip() and conf >= 0:
            words.append(word)
            confidences.append(conf)

    text = " ".join(words)
    mean_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

    return OCRResult(engine="tesseract", text=text, confidence=mean_confidence)
