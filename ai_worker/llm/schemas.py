from pydantic import BaseModel, Field


class HealthRiskFactor(BaseModel):
    name: str = Field(..., description="위험요인 이름")
    value: str | float | int | None = Field(default=None, description="위험요인 값")
    reason: str | None = Field(default=None, description="위험요인 설명")


class ChallengeRecommendation(BaseModel):
    name: str = Field(..., description="추천 챌린지 이름")
    reason: str | None = Field(default=None, description="추천 이유")


class DiseasePrediction(BaseModel):
    disease_name: str = Field(..., description="질환 이름")
    pred: int = Field(..., description="질환 위험 예측값")
    probability: float | None = Field(default=None, description="질환 위험 예측 확률")


class DiseasePredictionSet(BaseModel):
    hypertension: DiseasePrediction
    diabetes: DiseasePrediction
    dyslipidemia: DiseasePrediction
    obesity: DiseasePrediction


class RiskMappingResult(BaseModel):
    risk_group: str
    risk_factors: list[HealthRiskFactor] = Field(default_factory=list)
    recommended_challenges: list[ChallengeRecommendation] = Field(default_factory=list)


class RecommendationMessageInput(BaseModel):
    risk_factors: list[HealthRiskFactor]
    recommended_challenges: list[ChallengeRecommendation]
    tone: str = "friendly"


class RecommendationMessageOutput(BaseModel):
    summary_message: str
    challenge_message: str
    caution_message: str
    tone: str
    is_safe: bool = True


class ResultChatbotInput(BaseModel):
    user_message: str
    risk_factors: list[HealthRiskFactor] = Field(default_factory=list)
    recommended_challenges: list[ChallengeRecommendation] = Field(default_factory=list)
    tone: str = "friendly"


class ResultChatbotOutput(BaseModel):
    answer: str
    intent: str
    source: str
    referenced_health_factors: list[str] = Field(default_factory=list)
    recommended_challenges: list[ChallengeRecommendation] = Field(default_factory=list)
    caution_message: str
    tone: str
    is_safe: bool
    safety_result: dict


class MainHealthChatbotInput(BaseModel):
    user_message: str
    tone: str = "friendly"


class MainHealthChatbotOutput(BaseModel):
    answer: str
    intent: str
    source: str
    caution_message: str
    tone: str
    is_safe: bool
    safety_result: dict


HealthChatbotInput = ResultChatbotInput
HealthChatbotOutput = ResultChatbotOutput
