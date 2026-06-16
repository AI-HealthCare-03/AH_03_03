from tortoise import fields, models


class UserSetting(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.OneToOneField("models.User", related_name="settings")
    notification_enabled = fields.BooleanField(default=True)
    challenge_reminder_enabled = fields.BooleanField(default=True)
    challenge_reminder_time = fields.TimeField(null=True)
    medication_reminder_enabled = fields.BooleanField(default=True)
    diet_reminder_enabled = fields.BooleanField(default=False)
    marketing_agreed = fields.BooleanField(default=False)
    sensitive_data_agreed = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "user_settings"
