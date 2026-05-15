import json

from ai_worker.llm.llm_client import call_llm
from ai_worker.llm.prompt_templates import (
    RULE_BASED_MAIN_CHATBOT_REWRITE_PROMPT,
    RULE_BASED_RESULT_CHATBOT_REWRITE_PROMPT,
)
from ai_worker.llm.rule_engine import has_medical_consult_keyword
from ai_worker.llm.safety import check_medical_safety
from ai_worker.llm.schemas import (
    MainHealthChatbotInput,
    MainHealthChatbotOutput,
    ResultChatbotInput,
    ResultChatbotOutput,
)

CAUTION_MESSAGE = "이 정보는 진단이 아니며, 정확한 진단과 치료는 의료진 상담이 필요합니다."


def generate_result_chatbot_llm_response(
    input_data: ResultChatbotInput,
    use_real_llm: bool = False,
) -> ResultChatbotOutput:
    factor_names = [factor.name for factor in input_data.risk_factors]

    if use_real_llm:
        answer = call_llm(build_result_chatbot_prompt(input_data))
        source = "llm"
        intent = "llm_result_chatbot_response"
    else:
        answer = generate_result_chatbot_stub_answer(input_data)
        source = "llm_stub"
        intent = infer_result_llm_stub_intent(input_data.user_message)

    final_answer, safety_result = ensure_safe_answer(answer)

    return ResultChatbotOutput(
        answer=final_answer,
        intent=intent,
        source=source,
        referenced_health_factors=factor_names,
        recommended_challenges=input_data.recommended_challenges,
        caution_message=CAUTION_MESSAGE,
        tone=input_data.tone,
        is_safe=safety_result["is_safe"],
        safety_result=safety_result,
    )


def generate_main_health_chatbot_llm_response(
    input_data: MainHealthChatbotInput,
    use_real_llm: bool = False,
) -> MainHealthChatbotOutput:
    if use_real_llm:
        answer = call_llm(build_main_health_chatbot_prompt(input_data))
        source = "llm"
        intent = "llm_main_health_chatbot_response"
    else:
        answer = generate_main_health_chatbot_stub_answer(input_data)
        source = "llm_stub"
        intent = infer_main_llm_stub_intent(input_data.user_message)

    final_answer, safety_result = ensure_safe_answer(answer)

    return MainHealthChatbotOutput(
        answer=final_answer,
        intent=intent,
        source=source,
        caution_message=CAUTION_MESSAGE,
        tone=input_data.tone,
        is_safe=safety_result["is_safe"],
        safety_result=safety_result,
    )


def rewrite_result_chatbot_response_with_llm(
    input_data: ResultChatbotInput,
    rule_engine_output: ResultChatbotOutput,
    use_real_llm: bool = False,
) -> ResultChatbotOutput:
    if rule_engine_output.intent == "medical_consult_required":
        answer = rule_engine_output.answer
        source = "llm_rewrite_stub" if not use_real_llm else "llm_rewrite"
    elif use_real_llm:
        answer = call_llm_with_rewrite_fallback(
            prompt=build_result_chatbot_rewrite_prompt(input_data, rule_engine_output),
            fallback_answer=rule_engine_output.answer,
        )
        source = "llm_rewrite"
    else:
        answer = rewrite_answer_stub(rule_engine_output.answer)
        source = "llm_rewrite_stub"

    final_answer, safety_result = ensure_safe_answer(answer)

    # TODO: If grounding.py is added, call grounding checks here to verify the
    # rewritten answer did not add unsupported risk factors or challenges.
    return ResultChatbotOutput(
        answer=final_answer,
        intent=rule_engine_output.intent,
        source=source,
        referenced_health_factors=rule_engine_output.referenced_health_factors,
        recommended_challenges=rule_engine_output.recommended_challenges,
        caution_message=rule_engine_output.caution_message,
        tone=rule_engine_output.tone,
        is_safe=safety_result["is_safe"],
        safety_result=safety_result,
    )


def rewrite_main_health_chatbot_response_with_llm(
    input_data: MainHealthChatbotInput,
    rule_engine_output: MainHealthChatbotOutput,
    use_real_llm: bool = False,
) -> MainHealthChatbotOutput:
    if rule_engine_output.intent == "medical_consult_required":
        answer = rule_engine_output.answer
        source = "llm_rewrite_stub" if not use_real_llm else "llm_rewrite"
    elif use_real_llm:
        answer = call_llm_with_rewrite_fallback(
            prompt=build_main_health_chatbot_rewrite_prompt(input_data, rule_engine_output),
            fallback_answer=rule_engine_output.answer,
        )
        source = "llm_rewrite"
    else:
        answer = rewrite_answer_stub(rule_engine_output.answer)
        source = "llm_rewrite_stub"

    final_answer, safety_result = ensure_safe_answer(answer)

    return MainHealthChatbotOutput(
        answer=final_answer,
        intent=rule_engine_output.intent,
        source=source,
        caution_message=rule_engine_output.caution_message,
        tone=rule_engine_output.tone,
        is_safe=safety_result["is_safe"],
        safety_result=safety_result,
    )


def build_result_chatbot_prompt(input_data: ResultChatbotInput) -> str:
    factor_text = "\n".join(
        f"- {factor.name}: value={factor.value}, reason={factor.reason}" for factor in input_data.risk_factors
    )
    challenge_text = "\n".join(
        f"- {challenge.name}: reason={challenge.reason}" for challenge in input_data.recommended_challenges
    )

    return f"""
너는 만성질환 생활습관 관리 서비스의 결과 기반 챗봇이다.

규칙:
1. 진단, 확진, 치료, 처방, 약물 복용/중단 판단을 하지 않는다.
2. 입력된 건강정보 기준이라는 표현을 사용한다.
3. 추천 챌린지는 생활습관 관리 관점으로만 설명한다.
4. 약물/치료 질문은 의료진 상담을 권고한다.
5. 반드시 다음 의미를 포함한다: {CAUTION_MESSAGE}

사용자 질문:
{input_data.user_message}

위험요인:
{factor_text or "- 없음"}

추천 챌린지:
{challenge_text or "- 없음"}
""".strip()


def build_result_chatbot_rewrite_prompt(
    input_data: ResultChatbotInput,
    rule_engine_output: ResultChatbotOutput,
) -> str:
    factor_text = "\n".join(
        f"- {factor.name}: value={factor.value}, reason={factor.reason}" for factor in input_data.risk_factors
    )
    challenge_text = "\n".join(
        f"- {challenge.name}: reason={challenge.reason}" for challenge in input_data.recommended_challenges
    )

    return f"""
{RULE_BASED_RESULT_CHATBOT_REWRITE_PROMPT}

사용자 질문:
{input_data.user_message}

허용된 위험요인:
{factor_text or "- 없음"}

허용된 추천 챌린지:
{challenge_text or "- 없음"}

rule_engine_intent:
{rule_engine_output.intent}

rule_engine_answer:
{rule_engine_output.answer}
""".strip()


def build_main_health_chatbot_prompt(input_data: MainHealthChatbotInput) -> str:
    return f"""
너는 만성질환 생활습관 관리 서비스의 메인 건강 Q&A 챗봇이다.

규칙:
1. 진단, 확진, 치료, 처방, 약물 복용/중단 판단을 하지 않는다.
2. 고혈압, 당뇨, 이상지질혈증, 비만 관련 질문은 일반 생활습관 관리 관점에서 답한다.
3. 약물/치료 질문은 의료진 상담을 권고한다.
4. 반드시 다음 의미를 포함한다: {CAUTION_MESSAGE}

사용자 질문:
{input_data.user_message}
""".strip()


def build_main_health_chatbot_rewrite_prompt(
    input_data: MainHealthChatbotInput,
    rule_engine_output: MainHealthChatbotOutput,
) -> str:
    return f"""
{RULE_BASED_MAIN_CHATBOT_REWRITE_PROMPT}

사용자 질문:
{input_data.user_message}

rule_engine_intent:
{rule_engine_output.intent}

rule_engine_answer:
{rule_engine_output.answer}
""".strip()


def generate_result_chatbot_stub_answer(input_data: ResultChatbotInput) -> str:
    if has_medical_consult_keyword(input_data.user_message):
        return (
            "약 복용, 중단, 치료 여부는 개인의 검사 결과와 진료 이력을 함께 확인해야 하므로 "
            "의료진과 상담해 결정하는 것이 좋습니다."
        )

    factor_names = [factor.name for factor in input_data.risk_factors]
    challenge_names = [challenge.name for challenge in input_data.recommended_challenges]
    factor_text = ", ".join(factor_names) if factor_names else "입력된 건강정보"
    challenge_text = ", ".join(challenge_names) if challenge_names else "생활습관 관리"

    return (
        f"입력된 건강정보 기준으로 {factor_text} 항목을 함께 살펴볼 수 있습니다. "
        f"{challenge_text} 챌린지는 무리하지 않고 시작할 수 있는 생활습관 관리 방법입니다. "
        "처음에는 실천 가능한 작은 목표부터 꾸준히 이어가는 것이 도움이 될 수 있습니다."
    )


def generate_main_health_chatbot_stub_answer(input_data: MainHealthChatbotInput) -> str:
    user_message = input_data.user_message

    if has_medical_consult_keyword(user_message):
        return (
            "약 복용, 중단, 치료 여부는 일반 안내만으로 결정하기 어렵습니다. "
            "현재 상태와 진료 이력을 알고 있는 의료진과 상담해 결정하는 것이 좋습니다."
        )

    if any(keyword in user_message for keyword in ["당뇨", "혈당"]):
        return (
            "당뇨 관리는 혈당 변동을 줄이는 생활습관을 꾸준히 만드는 것이 중요합니다. "
            "단 음료와 정제 탄수화물을 줄이고, 식후 가벼운 걷기를 실천하는 것이 도움이 될 수 있습니다."
        )

    if any(keyword in user_message for keyword in ["고혈압", "혈압"]):
        return (
            "고혈압 관리는 혈관과 심장 부담을 줄이는 생활습관을 꾸준히 유지하는 것이 중요합니다. "
            "짠 음식 줄이기, 규칙적인 유산소 운동, 체중 관리가 도움이 될 수 있습니다."
        )

    return (
        "만성질환 예방에는 식사, 운동, 수면, 음주, 흡연, 스트레스 관리가 함께 영향을 줍니다. "
        "정기적인 건강검진과 꾸준한 생활습관 관리가 도움이 될 수 있습니다."
    )


def infer_result_llm_stub_intent(user_message: str) -> str:
    if has_medical_consult_keyword(user_message):
        return "medical_consult_required"
    if any(keyword in user_message for keyword in ["당뇨", "혈당"]):
        return "diabetes_result_guidance"
    if any(keyword in user_message for keyword in ["고혈압", "혈압"]):
        return "hypertension_result_guidance"
    return "llm_stub_result_guidance"


def infer_main_llm_stub_intent(user_message: str) -> str:
    if has_medical_consult_keyword(user_message):
        return "medical_consult_required"
    if any(keyword in user_message for keyword in ["당뇨", "혈당"]):
        return "diabetes_guidance"
    if any(keyword in user_message for keyword in ["고혈압", "혈압"]):
        return "hypertension_guidance"
    return "llm_stub_main_guidance"


def rewrite_answer_stub(rule_engine_answer: str) -> str:
    return rule_engine_answer.strip()


def call_llm_with_rewrite_fallback(prompt: str, fallback_answer: str) -> str:
    try:
        raw_response = call_llm(prompt)
        return extract_answer_from_json_response(raw_response)
    except Exception:
        return fallback_answer


def extract_answer_from_json_response(raw_response: str) -> str:
    parsed = json.loads(raw_response)
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("LLM rewrite response must include a non-empty answer field.")
    return answer


def ensure_safe_answer(answer: str) -> tuple[str, dict]:
    final_answer = answer.strip()
    if "진단이 아니" not in final_answer or "의료진 상담" not in final_answer:
        final_answer = f"{final_answer} {CAUTION_MESSAGE}"

    safety_result = check_medical_safety(final_answer)
    if safety_result["is_safe"]:
        return final_answer, safety_result

    safe_answer = (
        "건강 관련 판단은 입력된 정보만으로 확정하기 어렵습니다. "
        "생활습관 관리 방향은 참고용으로만 활용해 주세요. "
        f"{CAUTION_MESSAGE}"
    )
    safe_result = check_medical_safety(safe_answer)
    return safe_answer, safe_result
