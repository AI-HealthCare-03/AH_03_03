from app.dtos.llm_logs import LLMGenerationLogCreateRequest
from app.models.llm_logs import LLMGenerationLog
from app.repositories import llm_log_repository


async def create_llm_generation_log(user_id: int | None, request: LLMGenerationLogCreateRequest) -> LLMGenerationLog:
    return await llm_log_repository.create_llm_generation_log(user_id, request.model_dump())


async def get_llm_generation_log(log_id: int) -> LLMGenerationLog | None:
    return await llm_log_repository.get_llm_generation_log_by_id(log_id)


async def list_llm_generation_logs(
    user_id: int | None = None,
    target_type: str | None = None,
    target_id: int | None = None,
    llm_task_type: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[LLMGenerationLog]:
    return await llm_log_repository.list_llm_generation_logs(
        user_id=user_id,
        target_type=target_type,
        target_id=target_id,
        llm_task_type=llm_task_type,
        status=status,
        limit=limit,
        offset=offset,
    )
