from typing import Any

from app.models.llm_logs import LLMGenerationLog


async def create_llm_generation_log(user_id: int | None, data: dict[str, Any]) -> LLMGenerationLog:
    return await LLMGenerationLog.create(user_id=user_id, **data)


async def get_llm_generation_log_by_id(log_id: int) -> LLMGenerationLog | None:
    return await LLMGenerationLog.get_or_none(id=log_id)


async def list_llm_generation_logs(
    user_id: int | None = None,
    target_type: str | None = None,
    target_id: int | None = None,
    llm_task_type: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[LLMGenerationLog]:
    query = LLMGenerationLog.all()
    if user_id is not None:
        query = query.filter(user_id=user_id)
    if target_type is not None:
        query = query.filter(target_type=target_type)
    if target_id is not None:
        query = query.filter(target_id=target_id)
    if llm_task_type is not None:
        query = query.filter(llm_task_type=llm_task_type)
    if status is not None:
        query = query.filter(status=status)
    return await query.order_by("-created_at").offset(offset).limit(limit)


async def update_llm_generation_log(log_id: int, data: dict[str, Any]) -> LLMGenerationLog | None:
    log = await get_llm_generation_log_by_id(log_id)
    if log is None:
        return None
    for key, value in data.items():
        setattr(log, key, value)
    await log.save(update_fields=list(data.keys()) if data else None)
    return log


async def delete_llm_generation_log(log_id: int) -> int:
    return await LLMGenerationLog.filter(id=log_id).delete()
