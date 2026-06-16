from datetime import date

from pydantic import BaseModel


class TodayRecommendationItemResponse(BaseModel):
    title: str
    description: str
    reason: str
    action_type: str
    related_disease: str
    priority: int


class TodayRecommendationsResponse(BaseModel):
    date: date
    items: list[TodayRecommendationItemResponse]
