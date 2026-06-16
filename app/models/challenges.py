from enum import StrEnum

from tortoise import fields, models


class ChallengeCategory(StrEnum):
    EXERCISE = "EXERCISE"
    DIET = "DIET"
    SLEEP = "SLEEP"
    MEDICATION = "MEDICATION"
    WATER = "WATER"
    BLOOD_PRESSURE = "BLOOD_PRESSURE"
    BLOOD_GLUCOSE = "BLOOD_GLUCOSE"
    WEIGHT = "WEIGHT"
    HABIT = "HABIT"
    MONITORING = "MONITORING"
    MENTAL = "MENTAL"
    ORAL = "ORAL"


class ChallengeType(StrEnum):
    SPECIAL = "SPECIAL"
    COMMON = "COMMON"
    GENERAL = "GENERAL"
    ANEMIA = "ANEMIA"


class ChallengeTargetDisease(StrEnum):
    HYPERTENSION = "HYPERTENSION"
    DIABETES = "DIABETES"
    DYSLIPIDEMIA = "DYSLIPIDEMIA"
    OBESITY = "OBESITY"
    COMMON = "COMMON"
    GENERAL = "GENERAL"
    ANEMIA = "ANEMIA"


class ChallengeDifficulty(StrEnum):
    EASY = "EASY"
    NORMAL = "NORMAL"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class ChallengeStatus(StrEnum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class UserChallengeStatus(StrEnum):
    JOINED = "JOINED"
    COMPLETED = "COMPLETED"
    CANCELED = "CANCELED"


class Challenge(models.Model):
    id = fields.BigIntField(primary_key=True)
    title = fields.CharField(max_length=100)
    description = fields.TextField(null=True)
    category = fields.CharEnumField(enum_type=ChallengeCategory, max_length=20)
    challenge_type = fields.CharEnumField(enum_type=ChallengeType, max_length=10, default=ChallengeType.GENERAL)
    target_disease = fields.CharEnumField(
        enum_type=ChallengeTargetDisease,
        max_length=20,
        default=ChallengeTargetDisease.GENERAL,
    )
    difficulty = fields.CharEnumField(enum_type=ChallengeDifficulty, max_length=10, default=ChallengeDifficulty.NORMAL)
    target_metric = fields.CharField(max_length=100, null=True)
    target_value = fields.CharField(max_length=100, null=True)
    caution_message = fields.TextField(null=True)
    contraindication_message = fields.TextField(null=True)
    duration_days = fields.IntField()
    status = fields.CharEnumField(enum_type=ChallengeStatus, default=ChallengeStatus.ACTIVE)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "challenges"
        indexes = (("category", "status"), ("challenge_type", "target_disease", "status"))


class UserChallenge(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="user_challenges")
    challenge = fields.ForeignKeyField("models.Challenge", related_name="user_challenges")
    status = fields.CharEnumField(enum_type=UserChallengeStatus, default=UserChallengeStatus.JOINED)
    started_at = fields.DatetimeField()
    expected_done_at = fields.DatetimeField(null=True)
    completed_at = fields.DatetimeField(null=True)
    canceled_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "user_challenges"
        indexes = (("user_id", "status"), ("challenge_id",))


class ChallengeLog(models.Model):
    id = fields.BigIntField(primary_key=True)
    user_challenge = fields.ForeignKeyField("models.UserChallenge", related_name="logs")
    log_date = fields.DateField()
    is_completed = fields.BooleanField(default=False)
    completed_at = fields.DatetimeField(null=True)
    memo = fields.CharField(max_length=255, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "challenge_logs"
        indexes = (("user_challenge_id", "log_date"),)


class ChallengeRecommendation(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="challenge_recommendations")
    analysis_result = fields.ForeignKeyField("models.AnalysisResult", related_name="challenge_recommendations")
    challenge = fields.ForeignKeyField("models.Challenge", related_name="recommendations")
    reason = fields.TextField(null=True)
    priority = fields.IntField(default=0)
    is_selected = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "challenge_recommendations"
        indexes = (("user_id",), ("analysis_result_id",), ("challenge_id",), ("user_id", "is_selected"))
