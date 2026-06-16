from tortoise import fields, models


class FAQ(models.Model):
    id = fields.BigIntField(primary_key=True)
    category = fields.CharField(max_length=50)
    question = fields.CharField(max_length=255)
    answer = fields.TextField()
    display_order = fields.IntField(default=0)
    is_active = fields.BooleanField(default=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "faqs"
        indexes = (("category",), ("is_active", "display_order"))


class Inquiry(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="inquiries")
    category = fields.CharField(max_length=50)
    title = fields.CharField(max_length=255)
    content = fields.TextField()
    status = fields.CharField(max_length=30, default="PENDING")
    answer = fields.TextField(null=True)
    answered_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "inquiries"
        indexes = (("user_id",), ("status",), ("category",))
