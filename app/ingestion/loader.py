"""Multi-format document loader.

Accepts PDFs (native-text or scanned), and raster images (JPEG, PNG, TIFF).
For PDFs, native text extraction (PyMuPDF) is tried first. Whether a page
needs OCR is decided by text density: pages with very little extractable
text are almost certainly scans of images, not real text layers.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageSequence

from app.config import settings

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PDF_EXTENSIONS = {".pdf"}

RENDER_DPI = 200


@dataclass
class PageContent:
    page_number: int  # 1-indexed
    image: Image.Image
    native_text: str | None
    text_density: int  # character count of native text on this page
    needs_ocr: bool


@dataclass
class LoadedDocument:
    source_path: str
    mime_type: str
    pages: list[PageContent] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def is_fully_native_text(self) -> bool:
        """True if no page needs OCR (a clean, digitally-authored PDF)."""
        return all(not p.needs_ocr for p in self.pages)


def _needs_ocr(char_count: int) -> bool:
    return char_count < settings.ocr_min_text_density_chars_per_page


def load_pdf(path: str | Path) -> LoadedDocument:
    doc = fitz.open(str(path))
    pages: list[PageContent] = []

    for i, page in enumerate(doc):
        native_text = page.get_text("text") or ""
        char_count = len(native_text.strip())

        pixmap = page.get_pixmap(dpi=RENDER_DPI)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))

        pages.append(
            PageContent(
                page_number=i + 1,
                image=image,
                native_text=native_text if char_count > 0 else None,
                text_density=char_count,
                needs_ocr=_needs_ocr(char_count),
            )
        )

    doc.close()
    return LoadedDocument(source_path=str(path), mime_type="application/pdf", pages=pages)


def load_image(path: str | Path) -> LoadedDocument:
    path = Path(path)
    img = Image.open(path)
    pages: list[PageContent] = []

    frames = list(ImageSequence.Iterator(img)) if getattr(img, "n_frames", 1) > 1 else [img]

    for i, frame in enumerate(frames):
        rgb_frame = frame.convert("RGB")
        pages.append(
            PageContent(
                page_number=i + 1,
                image=rgb_frame,
                native_text=None,
                text_density=0,
                needs_ocr=True,
            )
        )

    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
    }.get(path.suffix.lower(), "application/octet-stream")

    return LoadedDocument(source_path=str(path), mime_type=mime_type, pages=pages)


def load_document(path: str | Path) -> LoadedDocument:
    """Dispatch to the right loader based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in PDF_EXTENSIONS:
        return load_pdf(path)
    if suffix in IMAGE_EXTENSIONS:
        return load_image(path)

    raise ValueError(
        f"Unsupported document format: {suffix!r}. "
        f"Supported: {sorted(PDF_EXTENSIONS | IMAGE_EXTENSIONS)}"
    )
