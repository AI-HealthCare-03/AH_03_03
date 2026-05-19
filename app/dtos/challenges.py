from datetime import date, datetime

from pydantic import BaseModel

from app.dtos.base import BaseSerializerModel
from app.models.challenges import ChallengeCategory, ChallengeStatus, UserChallengeStatus


class ChallengeCreateRequest(BaseModel):
    title: str
    description: str | None = None
    category: ChallengeCategory
    target_metric: str | None = None
    target_value: str | None = None
    duration_days: int
    status: ChallengeStatus = ChallengeStatus.ACTIVE


class ChallengeResponse(BaseSerializerModel):
    id: int
    title: str
    description: str | None
    category: ChallengeCategory
    target_metric: str | None
    target_value: str | None
    duration_days: int
    status: ChallengeStatus
    created_at: datetime
    updated_at: datetime


class UserChallengeCreateRequest(BaseModel):
    challenge_id: int
    status: UserChallengeStatus = UserChallengeStatus.JOINED
    started_at: datetime


class UserChallengeResponse(BaseSerializerModel):
    id: int
    user_id: int
    challenge_id: int
    status: UserChallengeStatus
    started_at: datetime
    completed_at: datetime | None
    canceled_at: datetime | None
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
