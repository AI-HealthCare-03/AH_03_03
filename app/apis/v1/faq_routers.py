from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.apis.v1.dependencies import (
    ensure_admin_user,
    ensure_found,
    ensure_owner_or_admin,
    get_request_user_with_firebase,
)
from app.dtos.faqs import FAQCreateRequest, FAQResponse, FAQUpdateRequest, InquiryCreateRequest, InquiryResponse
from app.models.users import User
from app.services import faqs as faq_service

faq_router = APIRouter(tags=["faqs"])


@faq_router.get("/faqs", response_model=list[FAQResponse])
async def list_active_faqs(category: str | None = None, limit: int = 20, offset: int = 0):
    return await faq_service.list_active_faqs(category=category, limit=limit, offset=offset)


@faq_router.post("/faqs", response_model=FAQResponse, status_code=status.HTTP_201_CREATED)
async def create_faq(request: FAQCreateRequest, user: Annotated[User, Depends(get_request_user_with_firebase)]):
    ensure_admin_user(user)
    return await faq_service.create_faq(request)


@faq_router.patch("/faqs/{faq_id}", response_model=FAQResponse)
async def update_faq(
    faq_id: int, request: FAQUpdateRequest, user: Annotated[User, Depends(get_request_user_with_firebase)]
):
    ensure_admin_user(user)
    ensure_found(await faq_service.get_faq(faq_id), "FAQ를 찾을 수 없습니다.")
    updated = await faq_service.update_faq(faq_id, request)
    return ensure_found(updated, "FAQ를 찾을 수 없습니다.")


@faq_router.post("/inquiries", response_model=InquiryResponse, status_code=status.HTTP_201_CREATED)
async def create_inquiry(request: InquiryCreateRequest, user: Annotated[User, Depends(get_request_user_with_firebase)]):
    return await faq_service.create_inquiry(user.id, request)


@faq_router.get("/inquiries/my", response_model=list[InquiryResponse])
async def list_user_inquiries(
    user: Annotated[User, Depends(get_request_user_with_firebase)], limit: int = 20, offset: int = 0
):
    return await faq_service.list_user_inquiries(user.id, limit=limit, offset=offset)


@faq_router.get("/inquiries/{inquiry_id}", response_model=InquiryResponse)
async def get_inquiry(inquiry_id: int, user: Annotated[User, Depends(get_request_user_with_firebase)]):
    inquiry = ensure_found(await faq_service.get_inquiry(inquiry_id), "문의글을 찾을 수 없습니다.")
    ensure_owner_or_admin(inquiry.user_id, user)
    return inquiry


@faq_router.patch("/inquiries/{inquiry_id}/answer", response_model=InquiryResponse)
async def answer_inquiry(inquiry_id: int, answer: str, user: Annotated[User, Depends(get_request_user_with_firebase)]):
    ensure_admin_user(user)
    ensure_found(await faq_service.get_inquiry(inquiry_id), "문의글을 찾을 수 없습니다.")
    answered = await faq_service.answer_inquiry(inquiry_id, answer)
    return ensure_found(answered, "문의글을 찾을 수 없습니다.")
