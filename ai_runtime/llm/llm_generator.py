import json

from ai_runtime.llm.grounding import check_result_chatbot_grounding
from ai_runtime.llm.llm_client import call_llm, call_llm_json, get_openai_model
from ai_runtime.llm.prompt_templates import (
    ANALYSIS_EXPLANATION_PROMPT_VERSION,
    HEALTH_CHAT_PROMPT_VERSION,
    MAIN_REWRITE_PROMPT_VERSION,
    RESULT_REWRITE_PROMPT_VERSION,
    RULE_BASED_MAIN_CHATBOT_REWRITE_PROMPT,
    RULE_BASED_RESULT_CHATBOT_REWRITE_PROMPT,
    render_prompt,
)
from ai_runtime.llm.rule_engine import has_medical_consult_keyword
from ai_runtime.llm.safety import check_medical_safety
from ai_runtime.llm.schemas import (
    MainHealthChatbotInput,
    MainHealthChatbotOutput,
    ResultChatbotInput,
    ResultChatbotOutput,
)

CAUTION_MESSAGE = "이 정보는 진단이 아니며 건강관리 참고용입니다. 정확한 진단과 치료는 의료진 상담이 필요합니다."


def generate_result_chatbot_llm_response(
    input_data: ResultChatbotInput,
    use_real_llm: bool = False,
) -> ResultChatbotOutput:
    factor_names = [factor.name for factor in input_data.risk_factors]

    if use_real_llm:
        answer = call_llm(
            build_result_chatbot_prompt(input_data),
            metadata={
                "prompt_version": ANALYSIS_EXPLANATION_PROMPT_VERSION,
                "source": "llm",
                "chatbot_type": "result_chatbot",
                "use_real_llm": True,
            },
        )
        source = "llm"
        intent = "llm_result_chatbot_response"
    else:
        answer = generate_result_chatbot_fallback_answer(input_data)
        source = "rule_based_llm_fallback"
        intent = infer_result_fallback_intent(input_data.user_message)

    final_answer, safety_result = ensure_safe_answer(answer)
    grounding_result = check_result_chatbot_grounding(
        answer=final_answer,
        allowed_factors=factor_names,
        allowed_challenges=[challenge.name for challenge in input_data.recommended_challenges],
        allowed_numbers=build_allowed_numbers_from_result_input(input_data),
        allow_numeric_values=False,
    )
    safety_result = merge_grounding_result(safety_result, grounding_result)
    safety_result = add_llm_metadata(
        safety_result,
        prompt_version=ANALYSIS_EXPLANATION_PROMPT_VERSION,
        model=get_openai_model() if use_real_llm else None,
    )

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
        answer = call_llm(
            build_main_health_chatbot_prompt(input_data),
            metadata={
                "prompt_version": HEALTH_CHAT_PROMPT_VERSION,
                "source": "llm",
                "chatbot_type": "main_health_chatbot",
                "use_real_llm": True,
            },
        )
        source = "llm"
        intent = "llm_main_health_chatbot_response"
    else:
        answer = generate_main_health_chatbot_fallback_answer(input_data)
        source = "rule_based_llm_fallback"
        intent = infer_main_fallback_intent(input_data.user_message)

    final_answer, safety_result = ensure_safe_answer(answer, include_caution_in_answer=False)
    safety_result = add_llm_metadata(
        safety_result,
        prompt_version=HEALTH_CHAT_PROMPT_VERSION,
        model=get_openai_model() if use_real_llm else None,
    )

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
        source = "rule_based_rewrite" if not use_real_llm else "llm_rewrite"
    elif use_real_llm:
        answer, llm_succeeded = call_llm_with_rewrite_fallback_status(
            prompt=build_result_chatbot_rewrite_prompt(input_data, rule_engine_output),
            fallback_answer=rule_engine_output.answer,
            metadata={
                "prompt_version": RESULT_REWRITE_PROMPT_VERSION,
                "source": "llm_rewrite",
                "chatbot_type": "result_chatbot",
                "use_real_llm": True,
            },
        )
        source = "openai_rewrite" if llm_succeeded else rule_engine_output.source
    else:
        answer = rewrite_result_chatbot_answer_fallback(
            input_data,
            rule_engine_output,
        )
        source = "rule_based_rewrite"

    final_answer, safety_result = ensure_safe_answer(answer)
    grounding_result = check_result_chatbot_grounding(
        answer=final_answer,
        allowed_factors=rule_engine_output.referenced_health_factors,
        allowed_challenges=[challenge.name for challenge in rule_engine_output.recommended_challenges],
        allowed_numbers=build_allowed_numbers_from_result_input(input_data),
        allow_numeric_values=False,
    )
    safety_result = merge_grounding_result(safety_result, grounding_result)
    safety_result = add_llm_metadata(
        safety_result,
        prompt_version=RESULT_REWRITE_PROMPT_VERSION,
        model=get_openai_model() if use_real_llm else None,
    )
    safety_result = add_rewrite_status_metadata(safety_result, use_real_llm=use_real_llm, source=source)

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
        source = "rule_based_rewrite" if not use_real_llm else "llm_rewrite"
    elif use_real_llm:
        answer, llm_succeeded = call_llm_with_rewrite_fallback_status(
            prompt=build_main_health_chatbot_rewrite_prompt(input_data, rule_engine_output),
            fallback_answer=rule_engine_output.answer,
            metadata={
                "prompt_version": MAIN_REWRITE_PROMPT_VERSION,
                "source": "llm_rewrite",
                "chatbot_type": "main_health_chatbot",
                "use_real_llm": True,
            },
        )
        source = "openai_rewrite" if llm_succeeded else rule_engine_output.source
    else:
        answer = rewrite_main_health_chatbot_answer_fallback(rule_engine_output)
        source = "rule_based_rewrite"

    final_answer, safety_result = ensure_safe_answer(answer, include_caution_in_answer=False)
    safety_result = add_llm_metadata(
        safety_result,
        prompt_version=MAIN_REWRITE_PROMPT_VERSION,
        model=get_openai_model() if use_real_llm else None,
    )
    safety_result = add_rewrite_status_metadata(safety_result, use_real_llm=use_real_llm, source=source)

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

    return render_prompt(
        "analysis_explanation_prompt",
        user_message=input_data.user_message,
        risk_factors=factor_text or "- 없음",
        recommended_challenges=challenge_text or "- 없음",
    )


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
    return render_prompt("health_chat_prompt", user_message=input_data.user_message)


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


def generate_result_chatbot_fallback_answer(input_data: ResultChatbotInput) -> str:
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


def generate_main_health_chatbot_fallback_answer(input_data: MainHealthChatbotInput) -> str:
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


def infer_result_fallback_intent(user_message: str) -> str:
    if has_medical_consult_keyword(user_message):
        return "medical_consult_required"
    if any(keyword in user_message for keyword in ["당뇨", "혈당"]):
        return "diabetes_result_guidance"
    if any(keyword in user_message for keyword in ["고혈압", "혈압"]):
        return "hypertension_result_guidance"
    return "rule_based_result_guidance"


def infer_main_fallback_intent(user_message: str) -> str:
    if has_medical_consult_keyword(user_message):
        return "medical_consult_required"
    if any(keyword in user_message for keyword in ["당뇨", "혈당"]):
        return "diabetes_guidance"
    if any(keyword in user_message for keyword in ["고혈압", "혈압"]):
        return "hypertension_guidance"
    return "rule_based_main_guidance"


def rewrite_result_chatbot_answer_fallback(
    input_data: ResultChatbotInput,
    rule_engine_output: ResultChatbotOutput,
) -> str:
    factor_names = [factor.name for factor in input_data.risk_factors]
    challenge_names = [challenge.name for challenge in input_data.recommended_challenges]

    factor_text = ", ".join(factor_names) if factor_names else "입력된 건강정보"
    challenge_text = ", ".join(challenge_names) if challenge_names else "생활습관 관리"

    if rule_engine_output.intent == "diabetes_result_guidance":
        return (
            f"입력된 건강정보 기준으로 {factor_text} 항목이 혈당 관리와 관련될 수 있습니다. "
            f"추천된 {challenge_text} 챌린지는 혈당 변동을 줄이는 생활습관 관리에 도움이 될 수 있습니다. "
            "처음에는 실천 가능한 작은 목표부터 꾸준히 이어가도 좋습니다."
        )

    if rule_engine_output.intent == "hypertension_result_guidance":
        return (
            f"입력된 건강정보 기준으로 {factor_text} 항목이 혈압 관리와 관련될 수 있습니다. "
            f"추천된 {challenge_text} 챌린지는 혈압 관리에 도움이 될 수 있는 생활습관입니다. "
            "처음에는 실천 가능한 작은 목표부터 꾸준히 이어가도 좋습니다."
        )

    return (
        f"입력된 건강정보 기준으로 {factor_text} 항목을 함께 살펴볼 수 있습니다. "
        f"추천된 {challenge_text} 챌린지는 건강 관리를 시작하는 데 도움이 될 수 있습니다. "
        "처음에는 실천 가능한 작은 목표부터 꾸준히 이어가도 좋습니다."
    )


def rewrite_main_health_chatbot_answer_fallback(
    rule_engine_output: MainHealthChatbotOutput,
) -> str:
    answer = remove_caution_message(rule_engine_output.answer)

    if rule_engine_output.intent == "diabetes_guidance":
        return (
            "당뇨와 관련된 생활습관은 한 번에 크게 바꾸기보다 꾸준히 이어갈 수 있는 방식으로 관리하는 것이 좋습니다. "
            f"{answer}"
        )

    if rule_engine_output.intent == "hypertension_guidance":
        return f"혈압 관리는 일상에서 반복되는 식사와 활동 습관을 조금씩 조정하는 것부터 시작할 수 있습니다. {answer}"

    if rule_engine_output.intent == "dyslipidemia_guidance":
        return f"지질 관리는 식사 구성과 신체활동을 함께 살펴보는 것이 도움이 될 수 있습니다. {answer}"

    if rule_engine_output.intent == "obesity_guidance":
        return f"체중 관리는 단기간의 변화보다 지속 가능한 습관을 만드는 방향으로 접근하는 것이 좋습니다. {answer}"

    return f"건강 관리는 작은 생활습관을 꾸준히 이어가는 것에서 시작할 수 있습니다. {answer}"


def remove_caution_message(answer: str) -> str:
    return answer.replace(CAUTION_MESSAGE, "").strip()


def build_allowed_numbers_from_result_input(input_data: ResultChatbotInput) -> list[str]:
    numbers: list[str] = []
    for factor in input_data.risk_factors:
        if factor.value is not None:
            numbers.append(str(factor.value))
    return numbers


def merge_grounding_result(safety_result: dict, grounding_result: dict) -> dict:
    merged = {
        **safety_result,
        "grounding_result": grounding_result,
    }
    merged["is_safe"] = bool(safety_result["is_safe"] and grounding_result["is_grounded"])
    return merged


def add_llm_metadata(
    safety_result: dict,
    prompt_version: str | None,
    model: str | None,
) -> dict:
    metadata = {
        **safety_result.get("metadata", {}),
        "prompt_version": prompt_version,
        "model": model,
    }
    return {
        **safety_result,
        "metadata": metadata,
    }


def add_rewrite_status_metadata(safety_result: dict, *, use_real_llm: bool, source: str) -> dict:
    metadata = {
        **safety_result.get("metadata", {}),
        "llm_requested": use_real_llm,
        "llm_used": source == "openai_rewrite",
        "source": source,
    }
    if use_real_llm and source != "openai_rewrite":
        metadata["fallback_used"] = True
        metadata["fallback_reason"] = "llm_rewrite_unavailable"
    return {
        **safety_result,
        "metadata": metadata,
    }


def call_llm_with_rewrite_fallback(
    prompt: str,
    fallback_answer: str,
    metadata: dict | None = None,
) -> str:
    answer, _ = call_llm_with_rewrite_fallback_status(prompt, fallback_answer, metadata)
    return answer


def call_llm_with_rewrite_fallback_status(
    prompt: str,
    fallback_answer: str,
    metadata: dict | None = None,
) -> tuple[str, bool]:
    try:
        raw_response = call_llm_json(
            prompt,
            schema_name="health_chatbot_rewrite",
            metadata=metadata,
        )
        return extract_answer_from_json_response(raw_response), True
    except Exception:
        return fallback_answer, False


def extract_answer_from_json_response(raw_response: str) -> str:
    parsed = json.loads(raw_response)
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("LLM rewrite response must include a non-empty answer field.")
    return answer


def ensure_safe_answer(answer: str, *, include_caution_in_answer: bool = True) -> tuple[str, dict]:
    final_answer = remove_caution_message(answer)
    if include_caution_in_answer and ("진단이 아니" not in final_answer or "의료진 상담" not in final_answer):
        final_answer = f"{final_answer} {CAUTION_MESSAGE}"

    safety_result = check_medical_safety(final_answer, require_disclaimer=include_caution_in_answer)
    if safety_result["is_safe"]:
        return final_answer, safety_result

    safe_answer = (
        "건강 관련 판단은 입력된 정보만으로 확정하기 어렵습니다. "
        "생활습관 관리 방향은 참고용으로만 활용해 주세요."
    )
    if include_caution_in_answer:
        safe_answer = f"{safe_answer} {CAUTION_MESSAGE}"
    safe_result = check_medical_safety(safe_answer, require_disclaimer=include_caution_in_answer)
    return safe_answer, safe_result
