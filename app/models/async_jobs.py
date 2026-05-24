from enum import StrEnum

from tortoise import fields, models


class AsyncJobStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class AsyncJob(models.Model):
    id = fields.BigIntField(primary_key=True)
    job_type = fields.CharField(max_length=50)
    status = fields.CharEnumField(enum_type=AsyncJobStatus, default=AsyncJobStatus.PENDING)
    request_payload = fields.JSONField(null=True)
    result_payload = fields.JSONField(null=True)
    error_message = fields.TextField(null=True)
    stream_id = fields.CharField(max_length=100, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    started_at = fields.DatetimeField(null=True)
    finished_at = fields.DatetimeField(null=True)

    class Meta:
        table = "async_jobs"
        indexes = (("job_type", "status"), ("status", "created_at"), ("stream_id",))
