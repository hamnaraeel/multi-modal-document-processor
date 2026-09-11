"""Image preprocessing for scanned pages: deskew, binarize, denoise, upscale.

Each step is applied conditionally based on a quick analysis of the image
(skew angle, contrast, resolution) rather than unconditionally, and every
step actually applied is recorded in a `PreprocessingReport` so the effect
on OCR confidence can be audited per page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from PIL import Image

MIN_DPI_EQUIVALENT_WIDTH = 1200  # below this, upscale before OCR
SKEW_ANGLE_THRESHOLD_DEGREES = 0.5


@dataclass
class PreprocessingReport:
    steps_applied: list[str] = field(default_factory=list)
    detected_skew_angle: float = 0.0
    original_size: tuple[int, int] = (0, 0)
    final_size: tuple[int, int] = (0, 0)
    ocr_confidence_before: float | None = None
    ocr_confidence_after: float | None = None

    def as_dict(self) -> dict:
        return {
            "steps_applied": self.steps_applied,
            "detected_skew_angle": round(self.detected_skew_angle, 3),
            "original_size": list(self.original_size),
            "final_size": list(self.final_size),
            "ocr_confidence_before": self.ocr_confidence_before,
            "ocr_confidence_after": self.ocr_confidence_after,
        }


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(mat: np.ndarray) -> Image.Image:
    if len(mat.shape) == 2:
        return Image.fromarray(mat)
    return Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))


def detect_skew_angle(gray: np.ndarray) -> float:
    """Estimate skew angle in degrees via minAreaRect over thresholded text pixels."""
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return float(angle)


def deskew(mat: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = mat.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        mat, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10)


def upscale(mat: np.ndarray, target_width: int = MIN_DPI_EQUIVALENT_WIDTH) -> np.ndarray:
    h, w = mat.shape[:2]
    if w >= target_width:
        return mat
    scale = target_width / w
    return cv2.resize(mat, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def auto_preprocess(image: Image.Image) -> tuple[Image.Image, PreprocessingReport]:
    """Apply only the preprocessing steps this image actually needs."""
    report = PreprocessingReport(original_size=image.size)

    mat = _pil_to_cv(image)
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

    # 1. Upscale low-resolution scans first (helps every later step).
    if mat.shape[1] < MIN_DPI_EQUIVALENT_WIDTH:
        mat = upscale(mat)
        gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        report.steps_applied.append("upscale")

    # 2. Deskew if the page is meaningfully rotated.
    angle = detect_skew_angle(gray)
    report.detected_skew_angle = angle
    if abs(angle) > SKEW_ANGLE_THRESHOLD_DEGREES:
        mat = deskew(mat, angle)
        gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        report.steps_applied.append("deskew")

    # 3. Denoise if the image is noisy (high local variance in flat regions).
    noise_estimate = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if noise_estimate > 500:
        gray = denoise(gray)
        report.steps_applied.append("denoise")

    # 4. Binarize for a cleaner OCR input.
    gray = binarize(gray)
    report.steps_applied.append("binarize")

    result = _cv_to_pil(gray)
    report.final_size = result.size
    return result, report


def preprocess_with_confidence_tracking(
    image: Image.Image,
    ocr_confidence_fn: Callable[[Image.Image], float],
) -> tuple[Image.Image, PreprocessingReport]:
    """Run OCR before/after preprocessing so we know whether it actually helped.

    `ocr_confidence_fn` should run OCR on the given image and return a 0-1
    confidence score. Whichever image (raw or preprocessed) scores higher is
    returned, so preprocessing never makes OCR quality worse for a given page.
    """
    report_before_confidence = ocr_confidence_fn(image)
    processed, report = auto_preprocess(image)
    report.ocr_confidence_before = report_before_confidence
    report.ocr_confidence_after = ocr_confidence_fn(processed)

    if report.ocr_confidence_after >= report.ocr_confidence_before:
        return processed, report

    report.steps_applied.append("reverted_to_original_higher_confidence")
    return image, report
