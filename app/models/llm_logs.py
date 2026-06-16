from tortoise import fields, models


class LLMGenerationLog(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField(
        "models.User", related_name="llm_generation_logs", null=True, on_delete=fields.SET_NULL
    )
    target_type = fields.CharField(max_length=50, null=True)
    target_id = fields.BigIntField(null=True)
    llm_task_type = fields.CharField(max_length=50)
    provider = fields.CharField(max_length=50, null=True)
    model_name = fields.CharField(max_length=100, null=True)
    prompt_version = fields.CharField(max_length=50, null=True)
    input_summary = fields.JSONField(null=True)
    output_text = fields.TextField(null=True)
    prompt_tokens = fields.IntField(null=True)
    completion_tokens = fields.IntField(null=True)
    total_tokens = fields.IntField(null=True)
    estimated_cost = fields.DecimalField(max_digits=12, decimal_places=6, null=True)
    status = fields.CharField(max_length=30, default="SUCCESS")
    error_message = fields.TextField(null=True)
    latency_ms = fields.IntField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "llm_generation_logs"
        indexes = (("user_id",), ("target_type", "target_id"), ("llm_task_type",), ("status",))
