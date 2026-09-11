"""Generate sample documents for demo purposes:

- sample_documents/invoice_native.pdf   -- a clean, digitally-authored
  invoice PDF with a real text layer (exercises the native-text path).
- sample_documents/invoice_scanned.png  -- the same invoice rendered as a
  plain image with slight rotation/noise, simulating a scan (exercises
  the OCR ensemble + preprocessing path).
- sample_documents/contract_native.pdf  -- a short sample service contract.

Run with: python scripts/generate_sample_documents.py
"""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_documents"

INVOICE_TEXT = """Acme Supplies Inc.
123 Market Street, Springfield

Invoice #INV-1042
Date: 2024-03-01      Due: 2024-03-31

Description          Qty   Unit Price   Total
Widget A               10        5.00      50.00
Widget B                5       12.00      60.00

Subtotal: 110.00
Tax: 8.80
Total: 118.80
Payment Terms: Net 30
"""

CONTRACT_TEXT = """SERVICE AGREEMENT

This Agreement is made effective as of January 1, 2024 between Acme Corp
("Client") and Beta LLC ("Provider").

Term: 12 months.

Provider shall deliver monthly reports to Client detailing service usage.

Either party may terminate this Agreement with 30 days written notice.

Governing law: State of Delaware.
"""


def make_native_pdf(text: str, out_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text, fontsize=11, fontname="helv")
    doc.save(str(out_path))
    doc.close()


def make_scanned_image(text: str, out_path: Path) -> None:
    img = Image.new("RGB", (1600, 2000), color="white")
    draw = ImageDraw.Image if False else None
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    draw.multiline_text((80, 80), text, fill="black", font=font, spacing=14)

    # Simulate scan artifacts: slight rotation + speckle noise.
    rotated = img.rotate(1.2, expand=True, fillcolor="white")
    arr = np.array(rotated).astype("int16")
    noise = np.random.randint(-12, 12, arr.shape, dtype="int16")
    noisy = np.clip(arr + noise, 0, 255).astype("uint8")
    Image.fromarray(noisy).save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_native_pdf(INVOICE_TEXT, OUT_DIR / "invoice_native.pdf")
    make_scanned_image(INVOICE_TEXT, OUT_DIR / "invoice_scanned.png")
    make_native_pdf(CONTRACT_TEXT, OUT_DIR / "contract_native.pdf")
    print(f"Sample documents written to {OUT_DIR}")


if __name__ == "__main__":
    main()
