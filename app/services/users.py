from datetime import datetime

from tortoise.transactions import in_transaction

from app.core import config
from app.core.utils.common import normalize_phone_number
from app.dtos.users import UserUpdateRequest
from app.models.users import User
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
        async with in_transaction():
            await self.repo.update_instance(
                user=user,
                data={
                    "is_active": False,
                    "deactivated_at": datetime.now(config.TIMEZONE),
                },
            )
            await user.refresh_from_db()
        return user
