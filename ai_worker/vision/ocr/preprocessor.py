"""
ai_worker/vision/ocr/preprocessor.py
건강검진표 이미지 품질 검사 및 전처리 모듈.
"""

import logging

import cv2
import numpy as np

from .schemas import QUALITY_GUIDE, ImageQualityReport, ImageQualityStatus

logger = logging.getLogger(__name__)

BLUR_THRESHOLD = 80.0
DARK_THRESHOLD = 60.0
BRIGHT_THRESHOLD = 220.0
SKEW_THRESHOLD = 5.0
MIN_WIDTH = 400
MIN_HEIGHT = 300


def bytes_to_cv2(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    return img


def check_blur(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_brightness(gray):
    return float(np.mean(gray))


def check_skew(gray):
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
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
    try:
        img = bytes_to_cv2(image_bytes)
    except ValueError:
        return ImageQualityReport(
            status=ImageQualityStatus.SMALL,
            message="이미지를 읽을 수 없습니다. 다시 촬영해주세요.",
            guide=QUALITY_GUIDE["small"]["guide"],
        )

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    img = bytes_to_cv2(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return binary
