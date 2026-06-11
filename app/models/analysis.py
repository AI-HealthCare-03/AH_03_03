from enum import StrEnum

from tortoise import fields, models


class AnalysisType(StrEnum):
    DIABETES = "DIABETES"
    OBESITY = "OBESITY"
    DYSLIPIDEMIA = "DYSLIPIDEMIA"
    HYPERTENSION = "HYPERTENSION"


class AnalysisMode(StrEnum):
    BASIC = "BASIC"
    PRECISION = "PRECISION"


class RiskLevel(StrEnum):
    LOW = "LOW"
    ATTENTION = "ATTENTION"
    CAUTION = "CAUTION"
    HIGH_CAUTION = "HIGH_CAUTION"


class FactorDirection(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class AnalysisResult(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="analysis_results")
    health_record = fields.ForeignKeyField("models.HealthRecord", related_name="analysis_results")
    async_job_id = fields.BigIntField(null=True)
    analysis_type = fields.CharEnumField(enum_type=AnalysisType)
    analysis_mode = fields.CharEnumField(enum_type=AnalysisMode, default=AnalysisMode.BASIC)
    risk_score = fields.DecimalField(max_digits=6, decimal_places=5)
    risk_level = fields.CharEnumField(enum_type=RiskLevel)
    summary = fields.CharField(max_length=255, null=True)
    model_name = fields.CharField(max_length=100, null=True)
    model_version = fields.CharField(max_length=50, null=True)
    analyzed_at = fields.DatetimeField()
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "analysis_results"
        indexes = (("user_id", "analyzed_at"), ("health_record_id", "analysis_type"))


class AnalysisResultFactor(models.Model):
    id = fields.BigIntField(primary_key=True)
    analysis_result = fields.ForeignKeyField("models.AnalysisResult", related_name="factors")
    factor_key = fields.CharField(max_length=100)
    factor_name = fields.CharField(max_length=100)
    factor_value = fields.CharField(max_length=100, null=True)
    contribution_score = fields.DecimalField(max_digits=10, decimal_places=6, null=True)
    direction = fields.CharEnumField(enum_type=FactorDirection)
    display_order = fields.IntField(default=0)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "analysis_result_factors"
        indexes = (("analysis_result_id", "display_order"),)


class AnalysisSnapshot(models.Model):
    id = fields.BigIntField(primary_key=True)
    analysis_result = fields.ForeignKeyField("models.AnalysisResult", related_name="snapshots")
    input_payload = fields.JSONField()
    output_payload = fields.JSONField()
    shap_payload = fields.JSONField(null=True)
    model_payload = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "analysis_snapshots"
