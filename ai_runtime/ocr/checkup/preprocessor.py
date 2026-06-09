"""
ai_runtime/ocr/checkup/preprocessor.py
건강검진표 이미지 품질 검사 및 전처리 모듈.
"""

import logging

import numpy as np

try:
    import cv2 as _cv2
except ModuleNotFoundError:
    _cv2 = None

from .schemas import QUALITY_GUIDE, ImageQualityReport, ImageQualityStatus

logger = logging.getLogger(__name__)

BLUR_THRESHOLD = 80.0
DARK_THRESHOLD = 60.0
BRIGHT_THRESHOLD = 220.0
SKEW_THRESHOLD = 5.0
MIN_WIDTH = 400
MIN_HEIGHT = 300
cv2 = _cv2


class OpenCvDependencyError(RuntimeError):
    """Raised when optional OpenCV OCR preprocessing dependencies are not installed."""


def _get_cv2_module():
    if cv2 is None:
        msg = "opencv-python is required for checkup OCR image preprocessing. Install OCR dependencies before running OCR."
        raise OpenCvDependencyError(msg)
    return cv2


def bytes_to_cv2(image_bytes):
    cv2_module = _get_cv2_module()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2_module.imdecode(arr, cv2_module.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    return img


def check_blur(gray):
    cv2_module = _get_cv2_module()
    return float(cv2_module.Laplacian(gray, cv2_module.CV_64F).var())


def check_brightness(gray):
    return float(np.mean(gray))


def check_skew(gray):
    cv2_module = _get_cv2_module()
    try:
        edges = cv2_module.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2_module.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return 0.0
        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if abs(angle) < 45:
                angles.append(angle)
        return float(np.median(angles)) if angles else 0.0
    except Exception:
        return 0.0


def assess_quality(image_bytes):
    cv2_module = _get_cv2_module()
    try:
        img = bytes_to_cv2(image_bytes)
    except ValueError:
        return ImageQualityReport(
            status=ImageQualityStatus.SMALL,
            message="이미지를 읽을 수 없습니다. 다시 촬영해주세요.",
            guide=QUALITY_GUIDE["small"]["guide"],
        )

    h, w = img.shape[:2]
    gray = cv2_module.cvtColor(img, cv2_module.COLOR_BGR2GRAY)

    blur_score = check_blur(gray)
    brightness = check_brightness(gray)
    skew_angle = check_skew(gray)

    logger.info("품질 검사 | blur=%.1f brightness=%.1f skew=%.1f size=%dx%d", blur_score, brightness, skew_angle, w, h)

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        q = QUALITY_GUIDE["small"]
        return ImageQualityReport(
            status=ImageQualityStatus.SMALL,
            message=q["message"],
            guide=q["guide"],
            blur_score=blur_score,
            brightness=brightness,
            skew_angle=skew_angle,
        )

    if blur_score < BLUR_THRESHOLD:
        q = QUALITY_GUIDE["blurry"]
        return ImageQualityReport(
            status=ImageQualityStatus.BLURRY,
            message=q["message"],
            guide=q["guide"],
            blur_score=blur_score,
            brightness=brightness,
            skew_angle=skew_angle,
        )

    if brightness < DARK_THRESHOLD or brightness > BRIGHT_THRESHOLD:
        q = QUALITY_GUIDE["dark"]
        return ImageQualityReport(
            status=ImageQualityStatus.DARK,
            message=q["message"],
            guide=q["guide"],
            blur_score=blur_score,
            brightness=brightness,
            skew_angle=skew_angle,
        )

    if abs(skew_angle) > SKEW_THRESHOLD:
        q = QUALITY_GUIDE["skewed"]
        return ImageQualityReport(
            status=ImageQualityStatus.SKEWED,
            message=q["message"],
            guide=q["guide"],
            blur_score=blur_score,
            brightness=brightness,
            skew_angle=skew_angle,
        )

    return ImageQualityReport(
        status=ImageQualityStatus.GOOD,
        message="이미지 품질이 양호합니다.",
        guide=[],
        blur_score=blur_score,
        brightness=brightness,
        skew_angle=skew_angle,
    )


def preprocess_for_ocr(image_bytes):
    cv2_module = _get_cv2_module()
    img = bytes_to_cv2(image_bytes)
    gray = cv2_module.cvtColor(img, cv2_module.COLOR_BGR2GRAY)
    denoised = cv2_module.fastNlMeansDenoising(gray, h=10)
    binary = cv2_module.adaptiveThreshold(
        denoised,
        255,
        cv2_module.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2_module.THRESH_BINARY,
        11,
        2,
    )
    return binary
