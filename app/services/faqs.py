from datetime import datetime

from app.core import config
from app.dtos.faqs import FAQCreateRequest, FAQUpdateRequest, InquiryCreateRequest, InquiryUpdateRequest
from app.models.faqs import FAQ, Inquiry
from app.repositories import faq_repository


async def create_faq(request: FAQCreateRequest) -> FAQ:
    return await faq_repository.create_faq(request.model_dump())


async def get_faq(faq_id: int) -> FAQ | None:
    return await faq_repository.get_faq_by_id(faq_id)


async def list_active_faqs(category: str | None = None, limit: int = 20, offset: int = 0) -> list[FAQ]:
    return await faq_repository.list_faqs(category=category, is_active=True, limit=limit, offset=offset)


async def update_faq(faq_id: int, request: FAQUpdateRequest) -> FAQ | None:
    return await faq_repository.update_faq(faq_id, request.model_dump(exclude_unset=True))


async def create_inquiry(user_id: int, request: InquiryCreateRequest) -> Inquiry:
    return await faq_repository.create_inquiry(user_id, request.model_dump())


async def get_inquiry(inquiry_id: int) -> Inquiry | None:
    return await faq_repository.get_inquiry_by_id(inquiry_id)


async def list_user_inquiries(user_id: int, limit: int = 20, offset: int = 0) -> list[Inquiry]:
    return await faq_repository.list_inquiries(user_id=user_id, limit=limit, offset=offset)


async def update_inquiry(inquiry_id: int, request: InquiryUpdateRequest) -> Inquiry | None:
    return await faq_repository.update_inquiry(inquiry_id, request.model_dump(exclude_unset=True))


async def answer_inquiry(inquiry_id: int, answer: str) -> Inquiry | None:
    data = {
        "answer": answer,
        "status": "ANSWERED",
        "answered_at": datetime.now(config.TIMEZONE),
    }
    return await faq_repository.update_inquiry(inquiry_id, data)
