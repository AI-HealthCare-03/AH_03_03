import secrets
from datetime import datetime

from tortoise.transactions import in_transaction

from app.core import config
from app.core.utils.common import normalize_phone_number
from app.core.utils.security import hash_password
from app.dtos.users import UserUpdateRequest
from app.models.analysis import AnalysisResult, AnalysisResultFactor, AnalysisSnapshot
from app.models.challenges import ChallengeLog, ChallengeRecommendation, UserChallenge
from app.models.diets import DietPhotoResult, DietRecord
from app.models.exams import ExamMeasurement, ExamReport
from app.models.family import FamilyMember, FamilyMemberStatus, FamilyNotificationSetting, FamilyShareSetting
from app.models.faqs import Inquiry
from app.models.health import HealthRecord
from app.models.llm_logs import LLMGenerationLog
from app.models.medications import Medication, MedicationRecord
from app.models.notifications import (
    Notification,
    NotificationLog,
    NotificationLogStatus,
    ReminderSchedule,
)
from app.models.rag import RAGRetrievalLog
from app.models.settings import UserSetting
from app.models.users import User, UserConsent
from app.repositories.user_repository import UserRepository
from app.services.auth import AuthService


class UserManageService:
    def __init__(self):
        self.repo = UserRepository()
        self.auth_service = AuthService()

    async def update_user(self, user: User, data: UserUpdateRequest) -> User:
        if data.email:
            await self.auth_service.check_email_exists(data.email)
        if data.login_id and data.login_id != user.login_id:
            await self.auth_service.check_login_id_exists(data.login_id)
        if data.phone_number:
            normalized_phone_number = normalize_phone_number(data.phone_number)
            await self.auth_service.check_phone_number_exists(normalized_phone_number)
            data.phone_number = normalized_phone_number
        async with in_transaction():
            await self.repo.update_instance(user=user, data=data.model_dump(exclude_none=True))
            await user.refresh_from_db()
        return user

    async def deactivate_user(self, user: User) -> User:
        now = datetime.now(config.TIMEZONE)
        original_email = user.email
        async with in_transaction():
            await self.repo.revoke_refresh_tokens_by_user(user.id, now)
            await self.repo.delete_password_reset_tokens_by_user(user.id)
            await self.repo.delete_verification_codes_by_email(original_email)
            await self._delete_sensitive_service_data(user.id)
            await self._disable_notification_delivery(user.id)
            await self._detach_family_links(user.id)
            await self._detach_llm_and_rag_logs(user.id)
            await self.repo.update_instance_allow_none(
                user=user,
                data={
                    "email": self._anonymized_email(user.id),
                    "login_id": None,
                    "phone_number": None,
                    "name": "탈퇴회원",
                    "nickname": None,
                    "address": None,
                    "profile_image_url": None,
                    "hashed_password": hash_password(secrets.token_urlsafe(32)),
                    "is_active": False,
                    "is_admin": False,
                    "role": "USER",
                    "failed_login_count": 0,
                    "locked_until": None,
                    "email_verified_at": None,
                    "deactivated_at": now,
                },
            )
            await user.refresh_from_db()
        return user

    def _anonymized_email(self, user_id: int) -> str:
        return f"deleted-{user_id}@deleted.local"

    async def _delete_sensitive_service_data(self, user_id: int) -> None:
        await AnalysisSnapshot.filter(analysis_result__user_id=user_id).delete()
        await AnalysisResultFactor.filter(analysis_result__user_id=user_id).delete()
        await ChallengeRecommendation.filter(user_id=user_id).delete()
        await AnalysisResult.filter(user_id=user_id).delete()
        await HealthRecord.filter(user_id=user_id).delete()

        await ExamMeasurement.filter(exam_report__user_id=user_id).delete()
        await ExamReport.filter(user_id=user_id).delete()

        await DietPhotoResult.filter(diet_record__user_id=user_id).delete()
        await DietRecord.filter(user_id=user_id).delete()

        await MedicationRecord.filter(user_id=user_id).delete()
        await Medication.filter(user_id=user_id).delete()

        await ChallengeLog.filter(user_challenge__user_id=user_id).delete()
        await UserChallenge.filter(user_id=user_id).delete()

        await Inquiry.filter(user_id=user_id).delete()
        await UserSetting.filter(user_id=user_id).delete()
        await UserConsent.filter(user_id=user_id).delete()

    async def _disable_notification_delivery(self, user_id: int) -> None:
        await ReminderSchedule.filter(user_id=user_id).update(is_active=False)
        await Notification.filter(user_id=user_id).delete()
        await NotificationLog.filter(user_id=user_id).update(
            title="탈퇴회원 알림",
            message_summary=None,
            error_message=None,
            status=NotificationLogStatus.CANCELED,
        )

    async def _detach_family_links(self, user_id: int) -> None:
        await FamilyMember.filter(user_id=user_id).update(
            user_id=None,
            display_name="탈퇴회원",
            phone_number=None,
            email=None,
            status=FamilyMemberStatus.REMOVED,
            is_registered=False,
        )
        await FamilyShareSetting.filter(owner_user_id=user_id).delete()
        await FamilyShareSetting.filter(viewer_user_id=user_id).delete()
        await FamilyNotificationSetting.filter(owner_user_id=user_id).delete()
        await FamilyNotificationSetting.filter(family_user_id=user_id).delete()

    async def _detach_llm_and_rag_logs(self, user_id: int) -> None:
        await LLMGenerationLog.filter(user_id=user_id).update(
            user_id=None,
            input_summary=None,
            output_text=None,
            error_message=None,
        )
        await RAGRetrievalLog.filter(user_id=user_id).update(
            user_id=None,
            query_text="[deleted-user]",
            retrieved_chunk_ids=None,
        )
