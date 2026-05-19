from typing import Any

from app.models.faqs import FAQ, Inquiry


async def create_faq(data: dict[str, Any]) -> FAQ:
    return await FAQ.create(**data)


async def get_faq_by_id(faq_id: int) -> FAQ | None:
    return await FAQ.get_or_none(id=faq_id)


async def list_faqs(
    category: str | None = None, is_active: bool | None = None, limit: int = 20, offset: int = 0
) -> list[FAQ]:
    query = FAQ.all()
    if category is not None:
        query = query.filter(category=category)
    if is_active is not None:
        query = query.filter(is_active=is_active)
    return await query.order_by("display_order", "-created_at").offset(offset).limit(limit)


async def update_faq(faq_id: int, data: dict[str, Any]) -> FAQ | None:
    faq = await get_faq_by_id(faq_id)
    if faq is None:
        return None
    for key, value in data.items():
        setattr(faq, key, value)
    await faq.save(update_fields=list(data.keys()) if data else None)
    return faq


async def delete_faq(faq_id: int) -> int:
    return await FAQ.filter(id=faq_id).delete()


async def create_inquiry(user_id: int, data: dict[str, Any]) -> Inquiry:
    return await Inquiry.create(user_id=user_id, **data)


async def get_inquiry_by_id(inquiry_id: int) -> Inquiry | None:
    return await Inquiry.get_or_none(id=inquiry_id)


async def list_inquiries(
    user_id: int | None = None,
    status: str | None = None,
    category: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[Inquiry]:
    query = Inquiry.all()
    if user_id is not None:
        query = query.filter(user_id=user_id)
    if status is not None:
        query = query.filter(status=status)
    if category is not None:
        query = query.filter(category=category)
    return await query.order_by("-created_at").offset(offset).limit(limit)


async def update_inquiry(inquiry_id: int, data: dict[str, Any]) -> Inquiry | None:
    inquiry = await get_inquiry_by_id(inquiry_id)
    if inquiry is None:
        return None
    for key, value in data.items():
        setattr(inquiry, key, value)
    await inquiry.save(update_fields=list(data.keys()) if data else None)
    return inquiry


async def delete_inquiry(inquiry_id: int) -> int:
    return await Inquiry.filter(id=inquiry_id).delete()
