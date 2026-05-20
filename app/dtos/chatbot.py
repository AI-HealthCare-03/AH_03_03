from enum import StrEnum

from pydantic import BaseModel


class ChatbotContextType(StrEnum):
    MAIN = "MAIN"
    ANALYSIS = "ANALYSIS"
    DIET = "DIET"
    CHALLENGE = "CHALLENGE"
    GENERAL = "GENERAL"


class ChatbotAskRequest(BaseModel):
    message: str
    context_type: ChatbotContextType = ChatbotContextType.GENERAL
    target_id: int | None = None


class ChatbotAskResponse(BaseModel):
    answer: str
    source: str = "DUMMY_LLM"
    context_type: ChatbotContextType
    recommended_actions: list[str]
    safety_notice: str
