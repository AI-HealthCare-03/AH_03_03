from __future__ import annotations

import re
from dataclasses import dataclass

STATIC_SAFETY_NOTICE = "본 서비스는 의료 진단이나 처방을 대신하지 않으며, 건강관리 참고 정보로 활용해 주세요."


@dataclass(frozen=True)
class StaticIntentResponse:
    intent: str
    source: str
    answer: str
    recommended_actions: list[str]


def match_static_intent(message: str | None) -> StaticIntentResponse | None:
    normalized = _normalize_message(message)
    if not normalized:
        return None
    if _is_greeting(normalized):
        return _greeting_response()
    if _is_service_intro_question(normalized):
        return _service_intro_response()
    if _is_help_question(normalized):
        return _help_response()
    return None


def _normalize_message(message: str | None) -> str:
    text = re.sub(r"\s+", " ", str(message or "")).strip().lower()
    return re.sub(r"[!?？。，.,~]+$", "", text).strip()


def _is_greeting(message: str) -> bool:
    return message in {
        "안녕하세요",
        "안녕",
        "안녕하세여",
        "하이",
        "ㅎㅇ",
        "hello",
        "hi",
        "hey",
    }


def _is_service_intro_question(message: str) -> bool:
    compact = message.replace(" ", "")
    if "healthladder" in compact and any(keyword in compact for keyword in ("뭐야", "무엇", "소개", "서비스")):
        return True
    return any(
        keyword in compact
        for keyword in (
            "이서비스가뭐야",
            "이서비스뭐야",
            "뭐하는서비스야",
            "무슨서비스야",
            "서비스소개",
            "헬스래더가뭐야",
            "헬스레더가뭐야",
            "헬스래더소개",
            "헬스레더소개",
        )
    )


def _is_help_question(message: str) -> bool:
    compact = message.replace(" ", "")
    return any(
        keyword in compact
        for keyword in (
            "뭐할수있어",
            "무엇을할수있어",
            "사용법",
            "도움말",
            "기능알려줘",
            "어떻게써",
            "어떻게사용",
            "사용방법",
        )
    )


def _greeting_response() -> StaticIntentResponse:
    return StaticIntentResponse(
        intent="greeting",
        source="static_greeting",
        answer=(
            "안녕하세요. Health Ladder입니다. 건강정보 기록, 검진표 분석, 식단 기록, 챌린지와 복약 관리에 "
            f"대해 궁금한 점을 물어보실 수 있어요. {STATIC_SAFETY_NOTICE}"
        ),
        recommended_actions=["건강정보 입력하기", "검진표 OCR 확인하기", "오늘의 건강관리 질문하기"],
    )


def _service_intro_response() -> StaticIntentResponse:
    return StaticIntentResponse(
        intent="service_intro",
        source="static_service_intro",
        answer=(
            "Health Ladder는 건강정보와 검진 결과를 기록하고, 위험도 분석 결과를 바탕으로 식단, 챌린지, "
            "복약 관리까지 이어서 볼 수 있게 돕는 건강관리 서비스입니다. "
            f"{STATIC_SAFETY_NOTICE}"
        ),
        recommended_actions=["건강 리포트 확인하기", "건강검진표 업로드하기", "챌린지 추천 보기"],
    )


def _help_response() -> StaticIntentResponse:
    return StaticIntentResponse(
        intent="help",
        source="static_help",
        answer=(
            "Health Ladder에서는 건강정보 입력, 검진표 OCR 확인, 간편/정밀 건강분석, 식단 기록과 추천, "
            "챌린지 실천, 복약 알림 설정을 사용할 수 있습니다. 궁금한 화면이나 기록을 기준으로 질문해 주세요. "
            f"{STATIC_SAFETY_NOTICE}"
        ),
        recommended_actions=["건강분석 실행하기", "식단 기록 남기기", "복약 알림 설정하기"],
    )
