from datetime import date, datetime, time

from tortoise.queryset import QuerySet

from app.core import config
from app.dtos.admin import (
    AdminSensitiveAccessLogResponse,
    AdminSummaryResponse,
    AdminSystemErrorLogResponse,
    AdminUsersSummaryResponse,
)
from app.models.analysis import AnalysisResult
from app.models.exams import ExamReport
from app.models.health import HealthRecord
from app.models.logs import SensitiveAccessLog, SystemErrorLog
from app.models.medications import Medication
from app.models.notifications import Notification
from app.models.users import User, UserRole
from app.services.email_service import EmailService


def _start_of_today() -> datetime:
    return datetime.combine(date.today(), time.min, tzinfo=config.TIMEZONE)


def _date_to_datetime(value: date | None, *, end_of_day: bool = False) -> datetime | None:
    if value is None:
        return None
    boundary = time.max if end_of_day else time.min
    return datetime.combine(value, boundary, tzinfo=config.TIMEZONE)


def _clamp_limit(limit: int) -> int:
    return min(max(limit, 1), 100)


class AdminMonitoringService:
    async def get_summary(self) -> AdminSummaryResponse:
        today_start = _start_of_today()

        return AdminSummaryResponse(
            total_users=await User.all().count(),
            active_users=await User.filter(is_active=True).count(),
            today_new_users=await User.filter(created_at__gte=today_start).count(),
            total_health_records=await HealthRecord.all().count(),
            total_analysis_results=await AnalysisResult.all().count(),
            total_exam_reports=await ExamReport.all().count(),
            total_medications=await Medication.all().count(),
            total_notifications=await Notification.all().count(),
            system_error_count_today=await SystemErrorLog.filter(created_at__gte=today_start).count(),
            sensitive_access_count_today=await SensitiveAccessLog.filter(created_at__gte=today_start).count(),
            email_service_status=EmailService().status(),
            environment=config.ENV,
        )

    async def get_users_summary(self) -> AdminUsersSummaryResponse:
        total_users = await User.all().count()
        active_users = await User.filter(is_active=True).count()
        today_new_users = await User.filter(created_at__gte=_start_of_today()).count()

        return AdminUsersSummaryResponse(
            total_users=total_users,
            active_users=active_users,
            inactive_users=max(total_users - active_users, 0),
            today_new_users=today_new_users,
            monitor_users=await User.filter(role=UserRole.MONITOR.value).count(),
            operator_users=await User.filter(role=UserRole.OPERATOR.value).count(),
            admin_users=await User.filter(role=UserRole.ADMIN.value).count(),
            super_admin_users=await User.filter(role=UserRole.SUPER_ADMIN.value).count(),
        )

    async def list_system_errors(
        self,
        *,
        limit: int = 50,
        date_from: date | None = None,
        date_to: date | None = None,
        status_code: int | None = None,
    ) -> tuple[list[AdminSystemErrorLogResponse], int, int]:
        safe_limit = _clamp_limit(limit)
        query = self._apply_date_filters(SystemErrorLog.all(), date_from=date_from, date_to=date_to)
        if status_code is not None:
            query = query.filter(status_code=status_code)

        total = await query.count()
        rows = await query.order_by("-created_at").limit(safe_limit)
        return (
            [
                AdminSystemErrorLogResponse(
                    id=int(row.id),
                    request_id=row.request_id,
                    user_id=row.user_id,
                    method=row.method,
                    path=row.path,
                    status_code=row.status_code,
                    error_type=row.error_type,
                    error_message=row.error_message,
                    client_ip=row.client_ip,
                    user_agent=row.user_agent,
                    created_at=row.created_at,
                )
                for row in rows
            ],
            total,
            safe_limit,
        )

    async def list_sensitive_access_logs(
        self,
        *,
        limit: int = 50,
        date_from: date | None = None,
        date_to: date | None = None,
        resource_type: str | None = None,
    ) -> tuple[list[AdminSensitiveAccessLogResponse], int, int]:
        safe_limit = _clamp_limit(limit)
        query = self._apply_date_filters(SensitiveAccessLog.all(), date_from=date_from, date_to=date_to)
        if resource_type:
            query = query.filter(resource_type=resource_type)

        total = await query.count()
        rows = await query.order_by("-created_at").limit(safe_limit)
        return (
            [
                AdminSensitiveAccessLogResponse(
                    id=int(row.id),
                    request_id=row.request_id,
                    actor_user_id=int(row.actor_user_id),
                    actor_role=row.actor_role,
                    target_user_id=int(row.target_user_id),
                    action_type=row.action_type,
                    resource_type=row.resource_type,
                    resource_id=row.resource_id,
                    access_reason=row.access_reason,
                    method=row.method,
                    path=row.path,
                    client_ip=row.client_ip,
                    user_agent=row.user_agent,
                    created_at=row.created_at,
                )
                for row in rows
            ],
            total,
            safe_limit,
        )

    @staticmethod
    def _apply_date_filters[T](
        query: QuerySet[T],
        *,
        date_from: date | None,
        date_to: date | None,
    ) -> QuerySet[T]:
        from_datetime = _date_to_datetime(date_from)
        to_datetime = _date_to_datetime(date_to, end_of_day=True)

        if from_datetime is not None:
            query = query.filter(created_at__gte=from_datetime)
        if to_datetime is not None:
            query = query.filter(created_at__lte=to_datetime)
        return query
