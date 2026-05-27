from tortoise import fields, models


class Medication(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="medications")
    name = fields.CharField(max_length=100)
    medication_type = fields.CharField(max_length=20)
    dosage = fields.CharField(max_length=100, null=True)
    frequency = fields.CharField(max_length=100, null=True)
    reminder_time = fields.TimeField(null=True)
    is_active = fields.BooleanField(default=True)
    memo = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "medications"
        indexes = (("user_id",), ("user_id", "is_active"), ("medication_type",))


class MedicationRecord(models.Model):
    id = fields.BigIntField(primary_key=True)
    medication = fields.ForeignKeyField("models.Medication", related_name="records")
    user = fields.ForeignKeyField("models.User", related_name="medication_records")
    scheduled_at = fields.DatetimeField(null=True)
    taken_at = fields.DatetimeField(null=True)
    is_taken = fields.BooleanField(default=False)
    status = fields.CharField(max_length=30, default="PENDING")
    memo = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "medication_records"
        indexes = (("medication_id",), ("user_id",), ("medication_id", "scheduled_at"), ("user_id", "scheduled_at"))
