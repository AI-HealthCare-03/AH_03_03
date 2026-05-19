from enum import StrEnum

from tortoise import fields, models


class OCRStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CONFIRMED = "CONFIRMED"


class ExamReport(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="exam_reports")
    original_filename = fields.CharField(max_length=255)
    file_path = fields.CharField(max_length=500)
    exam_date = fields.DateField(null=True)
    ocr_status = fields.CharEnumField(enum_type=OCRStatus, default=OCRStatus.PENDING)
    is_confirmed = fields.BooleanField(default=False)
    uploaded_at = fields.DatetimeField()
    confirmed_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "exam_reports"
        indexes = (("user_id", "uploaded_at"), ("user_id", "exam_date"), ("ocr_status",))


class ExamMeasurement(models.Model):
    id = fields.BigIntField(primary_key=True)
    exam_report = fields.ForeignKeyField("models.ExamReport", related_name="measurements")
    measurement_key = fields.CharField(max_length=100)
    measurement_name = fields.CharField(max_length=100)
    value = fields.CharField(max_length=100, null=True)
    unit = fields.CharField(max_length=30, null=True)
    ocr_confidence = fields.DecimalField(max_digits=5, decimal_places=4, null=True)
    is_user_confirmed = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "exam_measurements"
        indexes = (("exam_report_id",), ("measurement_key",))
