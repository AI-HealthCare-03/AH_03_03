from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

JPEG_MEDIA_TYPE = "image/jpeg"
PDF_MEDIA_TYPE = "application/pdf"
HEIC_MEDIA_TYPES = {"image/heic", "image/heif"}
HEIC_EXTENSIONS = {".heic", ".heif"}
PASSTHROUGH_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_UPLOAD_IMAGE_TYPES = PASSTHROUGH_IMAGE_TYPES | HEIC_MEDIA_TYPES
HEIC_CONVERSION_ERROR_MESSAGE = "HEIC 이미지를 처리하지 못했습니다. JPG/PNG로 다시 업로드해주세요."

_HEIF_OPENER_REGISTERED = False


@dataclass(frozen=True)
class NormalizedImage:
    data: bytes
    media_type: str
    converted: bool = False
    original_media_type: str | None = None
    original_filename: str | None = None


class ImageNormalizationError(ValueError):
    pass


def normalize_upload_image(
    image_bytes: bytes,
    media_type: str | None,
    filename: str | None = None,
) -> NormalizedImage:
    normalized_media_type = _normalize_media_type(media_type)
    if _is_pdf_upload(normalized_media_type, filename):
        return NormalizedImage(
            data=image_bytes,
            media_type=normalized_media_type or PDF_MEDIA_TYPE,
            original_media_type=media_type,
            original_filename=filename,
        )

    if not is_heic_upload(normalized_media_type, filename):
        return NormalizedImage(
            data=image_bytes,
            media_type=normalized_media_type or "application/octet-stream",
            original_media_type=media_type,
            original_filename=filename,
        )

    try:
        converted_bytes = _convert_heic_to_jpeg(image_bytes)
    except ImageNormalizationError:
        raise
    except Exception as exc:  # pragma: no cover - Pillow/plugin errors vary by platform.
        raise ImageNormalizationError(HEIC_CONVERSION_ERROR_MESSAGE) from exc

    return NormalizedImage(
        data=converted_bytes,
        media_type=JPEG_MEDIA_TYPE,
        converted=True,
        original_media_type=media_type,
        original_filename=filename,
    )


def is_heic_upload(media_type: str | None, filename: str | None = None) -> bool:
    normalized_media_type = _normalize_media_type(media_type)
    if normalized_media_type in HEIC_MEDIA_TYPES:
        return True
    return Path(filename or "").suffix.lower() in HEIC_EXTENSIONS


def _convert_heic_to_jpeg(image_bytes: bytes) -> bytes:
    _register_heif_opener_once()
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            output = BytesIO()
            image.save(output, format="JPEG", quality=95)
            return output.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageNormalizationError(HEIC_CONVERSION_ERROR_MESSAGE) from exc


def _register_heif_opener_once() -> None:
    global _HEIF_OPENER_REGISTERED
    if _HEIF_OPENER_REGISTERED:
        return
    register_heif_opener()
    _HEIF_OPENER_REGISTERED = True


def _normalize_media_type(media_type: str | None) -> str | None:
    if not media_type:
        return None
    return media_type.split(";", 1)[0].strip().lower()


def _is_pdf_upload(media_type: str | None, filename: str | None) -> bool:
    return media_type == PDF_MEDIA_TYPE or Path(filename or "").suffix.lower() == ".pdf"
