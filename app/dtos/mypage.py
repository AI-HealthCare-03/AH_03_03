from typing import Any

from pydantic import BaseModel


class MyPageSummaryResponse(BaseModel):
    user: Any
    settings: Any | None = None
    latest_health_record: Any | None = None
    unread_notification_count: int
