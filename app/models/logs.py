from tortoise import fields, models


class SystemErrorLog(models.Model):
    id = fields.BigIntField(primary_key=True)
    request_id = fields.CharField(max_length=64, null=True)
    user_id = fields.BigIntField(null=True)
    method = fields.CharField(max_length=10)
    path = fields.CharField(max_length=500)
    status_code = fields.IntField()
    error_type = fields.CharField(max_length=100)
    error_message = fields.TextField(null=True)
    stack_trace = fields.TextField(null=True)
    client_ip = fields.CharField(max_length=100, null=True)
    user_agent = fields.CharField(max_length=500, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "system_error_logs"
        indexes = (("request_id",), ("user_id",), ("status_code",), ("created_at",))


class SensitiveAccessLog(models.Model):
    id = fields.BigIntField(primary_key=True)
    request_id = fields.CharField(max_length=64, null=True)
    actor_user_id = fields.BigIntField()
    actor_role = fields.CharField(max_length=30, null=True)
    target_user_id = fields.BigIntField()
    action_type = fields.CharField(max_length=30)
    resource_type = fields.CharField(max_length=50)
    resource_id = fields.BigIntField(null=True)
    access_reason = fields.CharField(max_length=255, null=True)
    method = fields.CharField(max_length=10)
    path = fields.CharField(max_length=500)
    client_ip = fields.CharField(max_length=100, null=True)
    user_agent = fields.CharField(max_length=500, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "sensitive_access_logs"
        indexes = (
            ("request_id",),
            ("actor_user_id",),
            ("target_user_id",),
            ("resource_type",),
            ("created_at",),
        )
