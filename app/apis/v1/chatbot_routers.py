from typing import Annotated

from fastapi import APIRouter, Depends

from app.apis.v1.dependencies import get_request_user_with_firebase
from app.dtos.chatbot import ChatbotAskRequest, ChatbotAskResponse
from app.models.users import User
from app.services import chatbot as chatbot_service

chatbot_router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@chatbot_router.post("/ask", response_model=ChatbotAskResponse)
async def ask_chatbot(
    request: ChatbotAskRequest,
    user: Annotated[User, Depends(get_request_user_with_firebase)],
):
    _ = user
    return await chatbot_service.ask_dummy_chatbot(request)
