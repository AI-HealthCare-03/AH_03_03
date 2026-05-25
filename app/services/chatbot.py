from ai_worker.llm.response_router import route_main_health_chatbot_response
from ai_worker.llm.schemas import MainHealthChatbotInput
from app.dtos.chatbot import ChatbotAskRequest, ChatbotAskResponse, ChatbotContextType

SAFETY_NOTICE = "본 서비스는 진단/처방이 아니며, 치료 변경은 반드시 의료진과 상담해야 합니다."


async def ask_chatbot(request: ChatbotAskRequest) -> ChatbotAskResponse:
    # 공식 챗봇 API는 시연 안정성을 위해 OpenAI 호출 없이 로컬 룰엔진 경로를 사용한다.
    routed = route_main_health_chatbot_response(
        MainHealthChatbotInput(user_message=request.message),
        use_llm_fallback=False,
        use_llm_rewrite=False,
        use_real_llm=False,
    )
    actions = _recommended_actions(request)
    return ChatbotAskResponse(
        answer=routed.answer,
        source=routed.source,
        context_type=request.context_type,
        recommended_actions=actions,
        safety_notice=routed.caution_message,
    )


def _recommended_actions(request: ChatbotAskRequest) -> list[str]:
    message = request.message.lower()
    if request.context_type == ChatbotContextType.DIET or any(
        keyword in message for keyword in ["식단", "음식", "칼로리"]
    ):
        return ["오늘 식단 기록하기", "식단 분석 확인하기", "단 음료 대신 물 선택하기"]
    elif request.context_type == ChatbotContextType.CHALLENGE or any(
        keyword in message for keyword in ["운동", "걷기", "챌린지", "습관"]
    ):
        return ["챌린지 목록 보기", "오늘 완료 체크하기", "주 3회 걷기 목표 세우기"]
    elif any(keyword in message for keyword in ["혈당", "당뇨", "hba1c"]):
        return ["건강정보 입력", "건강 분석 실행하기", "식후 10분 산책 챌린지 참여"]
    elif any(keyword in message for keyword in ["혈압", "고혈압"]):
        return ["혈압 기록하기", "나트륨 줄이기", "수면 시간 점검"]
    elif any(keyword in message for keyword in ["약", "복약", "영양제"]):
        return ["복약 기록하기", "복약 알림 설정", "의료진 상담 메모 남기기"]
    return ["건강정보 입력", "분석 준비 상태 확인", "대시보드 보기"]
