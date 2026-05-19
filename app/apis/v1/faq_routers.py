from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.dependencies.security import get_request_user
from app.dtos.faqs import FAQCreateRequest, FAQResponse, FAQUpdateRequest, InquiryCreateRequest, InquiryResponse
from app.models.users import User
from app.services import faqs as faq_service

faq_router = APIRouter(tags=["faqs"])


@faq_router.get("/faqs", response_model=list[FAQResponse])
async def list_active_faqs(category: str | None = None, limit: int = 20, offset: int = 0):
    return await faq_service.list_active_faqs(category=category, limit=limit, offset=offset)


@faq_router.post("/faqs", response_model=FAQResponse, status_code=status.HTTP_201_CREATED)
async def create_faq(request: FAQCreateRequest):
    return await faq_service.create_faq(request)


@faq_router.patch("/faqs/{faq_id}", response_model=FAQResponse | None)
async def update_faq(faq_id: int, request: FAQUpdateRequest):
    return await faq_service.update_faq(faq_id, request)


@faq_router.post("/inquiries", response_model=InquiryResponse, status_code=status.HTTP_201_CREATED)
async def create_inquiry(request: InquiryCreateRequest, user: Annotated[User, Depends(get_request_user)]):
    return await faq_service.create_inquiry(user.id, request)


@faq_router.get("/inquiries/my", response_model=list[InquiryResponse])
async def list_user_inquiries(user: Annotated[User, Depends(get_request_user)], limit: int = 20, offset: int = 0):
    return await faq_service.list_user_inquiries(user.id, limit=limit, offset=offset)


@faq_router.get("/inquiries/{inquiry_id}", response_model=InquiryResponse | None)
async def get_inquiry(inquiry_id: int):
    return await faq_service.get_inquiry(inquiry_id)


@faq_router.patch("/inquiries/{inquiry_id}/answer", response_model=InquiryResponse | None)
async def answer_inquiry(inquiry_id: int, answer: str):
    return await faq_service.answer_inquiry(inquiry_id, answer)
