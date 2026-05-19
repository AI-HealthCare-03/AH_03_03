from tortoise import fields, models


class Notification(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="notifications")
    notification_type = fields.CharField(max_length=30)
    title = fields.CharField(max_length=100)
    message = fields.TextField()
    is_read = fields.BooleanField(default=False)
    related_type = fields.CharField(max_length=50, null=True)
    related_id = fields.BigIntField(null=True)
    read_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "notifications"
        indexes = (("user_id",), ("user_id", "is_read"), ("notification_type",))
