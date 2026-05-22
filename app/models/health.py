from tortoise import fields, models


class HealthRecord(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="health_records")
    height_cm = fields.DecimalField(max_digits=5, decimal_places=2, null=True)
    weight_kg = fields.DecimalField(max_digits=5, decimal_places=2, null=True)
    waist_cm = fields.DecimalField(max_digits=5, decimal_places=2, null=True)
    bmi = fields.DecimalField(max_digits=5, decimal_places=2, null=True)
    systolic_bp = fields.IntField(null=True)
    diastolic_bp = fields.IntField(null=True)
    fasting_glucose = fields.IntField(null=True)
    hba1c = fields.DecimalField(max_digits=4, decimal_places=2, null=True)
    total_cholesterol = fields.IntField(null=True)
    ldl_cholesterol = fields.IntField(null=True)
    hdl_cholesterol = fields.IntField(null=True)
    triglyceride = fields.IntField(null=True)
    has_diabetes = fields.BooleanField(null=True)
    has_obesity = fields.BooleanField(null=True)
    has_dyslipidemia = fields.BooleanField(null=True)
    has_hypertension = fields.BooleanField(null=True)
    is_smoker = fields.BooleanField(null=True)
    drinks_alcohol = fields.BooleanField(null=True)
    exercise_days_per_week = fields.IntField(null=True)
    sleep_hours = fields.DecimalField(max_digits=4, decimal_places=2, null=True)
    measured_at = fields.DatetimeField()
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "health_records"
        indexes = (("user_id", "measured_at"),)
