from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DemoJobCreateRequest(BaseModel):
    message: str = "hello"
    payload: dict[str, Any] | None = None


class AsyncJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_type: str
    status: str
    request_payload: dict[str, Any] | None
    result_payload: dict[str, Any] | None
    error_message: str | None
    stream_id: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
