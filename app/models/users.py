from enum import StrEnum

from tortoise import fields, models


class Gender(StrEnum):
    MALE = "MALE"
    FEMALE = "FEMALE"


class UserRole(StrEnum):
    USER = "USER"
    MONITOR = "MONITOR"
    OPERATOR = "OPERATOR"
    ADMIN = "ADMIN"
    SUPER_ADMIN = "SUPER_ADMIN"


class User(models.Model):
    id = fields.BigIntField(primary_key=True)
    login_id = fields.CharField(max_length=40, unique=True, null=True)
    email = fields.CharField(max_length=40)
    hashed_password = fields.CharField(max_length=255)
    name = fields.CharField(max_length=20)
    nickname = fields.CharField(max_length=30, null=True)
    gender = fields.CharEnumField(enum_type=Gender)
    birthday = fields.DateField()
    phone_number = fields.CharField(max_length=11, null=True)
    address = fields.CharField(max_length=255, null=True)
    profile_image_url = fields.CharField(max_length=500, null=True)
    role = fields.CharField(max_length=20, default="USER")
    is_active = fields.BooleanField(default=True)
    is_admin = fields.BooleanField(default=False)
    last_login_at = fields.DatetimeField(null=True)
    failed_login_count = fields.IntField(default=0)
    locked_until = fields.DatetimeField(null=True)
    deactivated_at = fields.DatetimeField(null=True)
    email_verified_at = fields.DatetimeField(null=True)
    privacy_consent_agreed_at = fields.DatetimeField(null=True)
    privacy_consent_version = fields.CharField(max_length=30, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "users"


class VerificationCode(models.Model):
    id = fields.BigIntField(primary_key=True)
    email = fields.CharField(max_length=40)
    code_hash = fields.CharField(max_length=128)
    purpose = fields.CharField(max_length=30, default="EMAIL_VERIFICATION")
    is_used = fields.BooleanField(default=False)
    expires_at = fields.DatetimeField()
    verified_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "verification_codes"


class PasswordResetToken(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="password_reset_tokens", on_delete=fields.CASCADE)
    token_hash = fields.CharField(max_length=128)
    is_used = fields.BooleanField(default=False)
    expires_at = fields.DatetimeField()
    used_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "password_reset_tokens"


class RefreshToken(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="refresh_tokens", on_delete=fields.CASCADE)
    token_jti = fields.CharField(max_length=64, unique=True)
    expires_at = fields.DatetimeField()
    revoked_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "refresh_tokens"


class UserConsent(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="consents", on_delete=fields.CASCADE)
    terms_agreed = fields.BooleanField(default=True)
    privacy_agreed = fields.BooleanField(default=True)
    sensitive_data_agreed = fields.BooleanField(default=False)
    marketing_agreed = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "user_consents"
