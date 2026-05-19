from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict


class LLMGenerationLogCreateRequest(BaseModel):
    target_type: str | None = None
    target_id: int | None = None
    llm_task_type: str
    provider: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    input_summary: dict[str, Any] | None = None
    output_text: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: Decimal | None = None
    status: str = "SUCCESS"
    error_message: str | None = None
    latency_ms: int | None = None


class LLMGenerationLogUpdateRequest(BaseModel):
    target_type: str | None = None
    target_id: int | None = None
    llm_task_type: str | None = None
    provider: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    input_summary: dict[str, Any] | None = None
    output_text: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: Decimal | None = None
    status: str | None = None
    error_message: str | None = None
    latency_ms: int | None = None


class LLMGenerationLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int | None
    target_type: str | None
    target_id: int | None
    llm_task_type: str
    provider: str | None
    model_name: str | None
    prompt_version: str | None
    input_summary: dict[str, Any] | None
    output_text: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost: Decimal | None
    status: str
    error_message: str | None
    latency_ms: int | None
    created_at: datetime


class LLMGenerationLogListResponse(BaseModel):
    items: list[LLMGenerationLogResponse]
    total: int
