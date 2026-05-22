from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FAQCreateRequest(BaseModel):
    category: str
    question: str
    answer: str
    display_order: int = 0
    is_active: bool = True


class FAQUpdateRequest(BaseModel):
    category: str | None = None
    question: str | None = None
    answer: str | None = None
    display_order: int | None = None
    is_active: bool | None = None


class FAQResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    category: str
    question: str
    answer: str
    display_order: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class FAQListResponse(BaseModel):
    items: list[FAQResponse]
    total: int


class InquiryCreateRequest(BaseModel):
    category: str
    title: str
    content: str


class InquiryUpdateRequest(BaseModel):
    category: str | None = None
    title: str | None = None
    content: str | None = None
    status: str | None = None
    answer: str | None = None
    answered_at: datetime | None = None


class InquiryAnswerRequest(BaseModel):
    answer: str


class InquiryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    category: str
    title: str
    content: str
    status: str
    answer: str | None
    answered_at: datetime | None
    created_at: datetime
    updated_at: datetime


class InquiryListResponse(BaseModel):
    items: list[InquiryResponse]
    total: int
