from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.dtos.base import BaseSerializerModel
from app.models.challenges import (
    ChallengeCategory,
    ChallengeDifficulty,
    ChallengeStatus,
    ChallengeTargetDisease,
    ChallengeType,
    UserChallengeStatus,
)


class ChallengeCreateRequest(BaseModel):
    title: str
    description: str | None = None
    category: ChallengeCategory
    challenge_type: ChallengeType = ChallengeType.GENERAL
    target_disease: ChallengeTargetDisease = ChallengeTargetDisease.GENERAL
    difficulty: ChallengeDifficulty = ChallengeDifficulty.NORMAL
    target_metric: str | None = None
    target_value: str | None = None
    caution_message: str | None = None
    contraindication_message: str | None = None
    duration_days: int
    status: ChallengeStatus = ChallengeStatus.ACTIVE


class ChallengeResponse(BaseSerializerModel):
    id: int
    title: str
    description: str | None
    category: ChallengeCategory
    challenge_type: ChallengeType
    target_disease: ChallengeTargetDisease
    difficulty: ChallengeDifficulty
    target_metric: str | None
    target_value: str | None
    caution_message: str | None
    contraindication_message: str | None
    duration_days: int
    status: ChallengeStatus
    created_at: datetime
    updated_at: datetime


class UserChallengeCreateRequest(BaseModel):
    challenge_id: int
    status: UserChallengeStatus = UserChallengeStatus.JOINED
    started_at: datetime


class UserChallengeUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: UserChallengeStatus | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    canceled_at: datetime | None = None


class UserChallengeResponse(BaseSerializerModel):
    id: int
    user_id: int
    challenge_id: int
    status: UserChallengeStatus
    started_at: datetime
    completed_at: datetime | None
    canceled_at: datetime | None
    completed_days: int = 0
    progress: int = 0
    today_completed: bool = False
    today_completed_count: int = 0
    daily_goal_count: int = 1
    duration_days: int | None = None
    created_at: datetime
    updated_at: datetime


class ChallengeLogCreateRequest(BaseModel):
    log_date: date
    is_completed: bool = False
    memo: str | None = None


class ChallengeLogResponse(BaseSerializerModel):
    id: int
    user_challenge_id: int
    log_date: date
    is_completed: bool
    memo: str | None
    created_at: datetime
    updated_at: datetime


class ChallengeRecommendationCreateRequest(BaseModel):
    challenge_id: int
    analysis_result_id: int | None = None
    reason: str | None = None
    priority: int = 0
    is_selected: bool = False


class ChallengeRecommendationResponse(BaseSerializerModel):
    id: int
    user_id: int
    analysis_result_id: int | None
    challenge_id: int
    reason: str | None
    priority: int
    is_selected: bool
    created_at: datetime


class ChallengeActionResponse(BaseModel):
    message: str
    result: Any
