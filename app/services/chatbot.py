from app.dtos.chatbot import ChatbotAskRequest, ChatbotAskResponse, ChatbotContextType

SAFETY_NOTICE = "본 서비스는 진단/처방이 아니며, 치료 변경은 반드시 의료진과 상담해야 합니다."


async def ask_chatbot(request: ChatbotAskRequest) -> ChatbotAskResponse:
    # TODO: route through ai_worker.llm response router when the production LLM path is enabled.
    message = request.message.lower()
    if request.context_type == ChatbotContextType.DIET or any(
        keyword in message for keyword in ["식단", "음식", "칼로리"]
    ):
        answer = "식단은 당류 음료와 야식을 줄이고, 단백질과 채소를 함께 챙기는 방향으로 시작해 보세요."
        actions = ["오늘 식단 기록하기", "식단 분석 확인하기", "단 음료 대신 물 선택하기"]
    elif request.context_type == ChatbotContextType.CHALLENGE or any(
        keyword in message for keyword in ["운동", "걷기", "챌린지", "습관"]
    ):
        answer = "처음에는 짧은 챌린지를 고르는 게 좋습니다. 식후 10분 산책처럼 바로 실행 가능한 목표를 추천합니다."
        actions = ["챌린지 목록 보기", "오늘 완료 체크하기", "주 3회 걷기 목표 세우기"]
    elif any(keyword in message for keyword in ["혈당", "당뇨", "hba1c"]):
        answer = "혈당 관리는 공복혈당과 식후 활동 기록을 함께 보는 것이 좋습니다. 최근 수치를 꾸준히 입력해 보세요."
        actions = ["건강정보 입력", "건강 분석 실행하기", "식후 10분 산책 챌린지 참여"]
    elif any(keyword in message for keyword in ["혈압", "고혈압"]):
        answer = "혈압은 같은 시간대에 반복 측정한 기록이 중요합니다. 나트륨 섭취와 수면도 함께 확인해 보세요."
        actions = ["혈압 기록하기", "나트륨 줄이기", "수면 시간 점검"]
    elif any(keyword in message for keyword in ["약", "복약", "영양제"]):
        answer = "복약 정보는 누락 없이 기록하되, 약 복용 변경이나 중단은 의료진과 먼저 상담해야 합니다."
        actions = ["복약 기록하기", "복약 알림 설정", "의료진 상담 메모 남기기"]
    else:
        answer = "건강정보를 입력하면 당뇨, 비만, 이상지질혈증 중심의 건강 분석 결과를 확인할 수 있습니다."
        actions = ["건강정보 입력", "분석 준비 상태 확인", "대시보드 보기"]

    return ChatbotAskResponse(
        answer=answer,
        context_type=request.context_type,
        recommended_actions=actions,
        safety_notice=SAFETY_NOTICE,
    )
