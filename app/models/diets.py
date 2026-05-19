from tortoise import fields, models


class DietRecord(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="diet_records")
    meal_type = fields.CharField(max_length=20, null=True)
    meal_time = fields.DatetimeField(null=True)
    description = fields.TextField(null=True)
    image_path = fields.CharField(max_length=500, null=True)
    detected_foods = fields.JSONField(null=True)
    nutrition_summary = fields.JSONField(null=True)
    diet_score = fields.FloatField(null=True)
    diet_feedback = fields.TextField(null=True)
    analysis_method = fields.CharField(max_length=30, null=True)
    is_user_corrected = fields.BooleanField(default=False)
    memo = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "diet_records"
        indexes = (("user_id",), ("user_id", "meal_time"), ("analysis_method",))


class DietPhotoResult(models.Model):
    id = fields.BigIntField(primary_key=True)
    diet_record = fields.ForeignKeyField("models.DietRecord", related_name="photo_results")
    detected_foods = fields.JSONField(null=True)
    confidence_payload = fields.JSONField(null=True)
    raw_output = fields.JSONField(null=True)
    is_dummy = fields.BooleanField(default=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "diet_photo_results"
        indexes = (("diet_record_id",),)
