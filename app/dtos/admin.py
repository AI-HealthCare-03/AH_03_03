from datetime import datetime

from pydantic import BaseModel, Field


class AdminSummaryResponse(BaseModel):
    total_users: int
    active_users: int
    today_new_users: int
    total_health_records: int
    total_analysis_results: int
    total_exam_reports: int
    total_medications: int
    total_notifications: int
    system_error_count_today: int
    sensitive_access_count_today: int
    email_service_status: str
    environment: str


class AdminUsersSummaryResponse(BaseModel):
    total_users: int
    active_users: int
    inactive_users: int
    today_new_users: int
    monitor_users: int
    operator_users: int
    admin_users: int
    super_admin_users: int


class AdminSystemHealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    checks: dict[str, str]
    details: dict[str, str] = Field(default_factory=dict)


class AdminSystemErrorLogResponse(BaseModel):
    id: int
    request_id: str | None
    user_id: int | None
    method: str
    path: str
    status_code: int
    error_type: str
    error_message: str | None
    client_ip: str | None
    user_agent: str | None
    created_at: datetime


class AdminSensitiveAccessLogResponse(BaseModel):
    id: int
    request_id: str | None
    actor_user_id: int
    actor_role: str | None
    target_user_id: int
    action_type: str
    resource_type: str
    resource_id: int | None
    access_reason: str | None
    method: str
    path: str
    client_ip: str | None
    user_agent: str | None
    created_at: datetime


class AdminSystemErrorLogListResponse(BaseModel):
    items: list[AdminSystemErrorLogResponse]
    total: int
    limit: int
    filters: dict[str, str | int | None]


class AdminSensitiveAccessLogListResponse(BaseModel):
    items: list[AdminSensitiveAccessLogResponse]
    total: int
    limit: int
    filters: dict[str, str | int | None]
