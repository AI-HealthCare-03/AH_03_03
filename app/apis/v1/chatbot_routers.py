import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.apis.v1.dependencies import get_request_user
from app.core import config
from app.dtos.chatbot import ChatbotAskRequest, ChatbotAskResponse
from app.models.users import User
from app.services import chatbot as chatbot_service

chatbot_router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@chatbot_router.post("/ask", response_model=ChatbotAskResponse)
async def ask_chatbot(
    request: Request,
    response: Response,
    payload: ChatbotAskRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    _ = user
    chatbot_response, runtime_trace = await chatbot_service.ask_chatbot_with_runtime_trace(payload)
    if _should_include_smoke_trace_header(request) and runtime_trace:
        response.headers["X-Chatbot-Rag-Trace"] = json.dumps(runtime_trace, ensure_ascii=False, separators=(",", ":"))
    return chatbot_response


def _should_include_smoke_trace_header(request: Request) -> bool:
    if request.headers.get("x-chatbot-smoke-trace", "").lower() != "true":
        return False
    env = getattr(config, "ENV", "")
    env_value = getattr(env, "value", str(env))
    return str(env_value).lower() in {"local", "dev"}
