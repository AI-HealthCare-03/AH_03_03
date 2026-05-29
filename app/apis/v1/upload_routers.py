from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response

from ai_runtime.common.image_normalizer import (
    HEIC_CONVERSION_ERROR_MESSAGE,
    PDF_MEDIA_TYPE,
    ImageNormalizationError,
    normalize_upload_image,
)
from app.apis.v1.dependencies import get_request_user
from app.models.users import User

upload_router = APIRouter(prefix="/uploads", tags=["uploads"])


@upload_router.post("/normalize-image", response_class=Response)
async def normalize_image_for_preview(
    user: Annotated[User, Depends(get_request_user)],
    file: Annotated[UploadFile, File(...)],
) -> Response:
    image_bytes = await file.read()
    normalized_image = _normalize_preview_image(image_bytes, file.content_type, file.filename)
    return Response(content=normalized_image.data, media_type=normalized_image.media_type)


def _normalize_preview_image(image_bytes: bytes, media_type: str | None, filename: str | None):
    if _is_pdf_upload(media_type, filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="PDF는 이미지 미리보기 변환 대상이 아닙니다.",
        )

    try:
        return normalize_upload_image(image_bytes, media_type, filename)
    except ImageNormalizationError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc) or HEIC_CONVERSION_ERROR_MESSAGE,
        ) from exc


def _is_pdf_upload(media_type: str | None, filename: str | None) -> bool:
    normalized_media_type = (media_type or "").split(";", 1)[0].strip().lower()
    return normalized_media_type == PDF_MEDIA_TYPE or (filename or "").lower().endswith(".pdf")
