from ai_worker.llm.health_chatbot import (
    generate_main_health_chatbot_response,
    generate_result_chatbot_response,
)
from ai_worker.llm.llm_generator import (
    generate_main_health_chatbot_llm_response,
    generate_result_chatbot_llm_response,
    rewrite_main_health_chatbot_response_with_llm,
    rewrite_result_chatbot_response_with_llm,
)
from ai_worker.llm.schemas import (
    MainHealthChatbotInput,
    MainHealthChatbotOutput,
    ResultChatbotInput,
    ResultChatbotOutput,
)


def route_result_chatbot_response(
    input_data: ResultChatbotInput,
    use_llm_fallback: bool = False,
    use_llm_rewrite: bool = False,
    use_real_llm: bool = False,
) -> ResultChatbotOutput:
    rule_result = generate_result_chatbot_response(input_data)
    if use_llm_rewrite:
        if rule_result.intent == "medical_consult_required":
            return rule_result

        rewrite_result = rewrite_result_chatbot_response_with_llm(
            input_data,
            rule_engine_output=rule_result,
            use_real_llm=use_real_llm,
        )
        if rewrite_result.is_safe:
            return rewrite_result
        return with_fallback_metadata(rule_result, rewrite_result)

    if rule_result.source == "rule_engine":
        return rule_result

    if not use_llm_fallback:
        return rule_result

    return generate_result_chatbot_llm_response(
        input_data,
        use_real_llm=use_real_llm,
    )


def route_main_health_chatbot_response(
    input_data: MainHealthChatbotInput,
    use_llm_fallback: bool = False,
    use_llm_rewrite: bool = False,
    use_real_llm: bool = False,
) -> MainHealthChatbotOutput:
    rule_result = generate_main_health_chatbot_response(input_data)
    if use_llm_rewrite:
        if rule_result.intent == "medical_consult_required":
            return rule_result

        rewrite_result = rewrite_main_health_chatbot_response_with_llm(
            input_data,
            rule_engine_output=rule_result,
            use_real_llm=use_real_llm,
        )
        if rewrite_result.is_safe:
            return rewrite_result
        return with_fallback_metadata(rule_result, rewrite_result)

    if rule_result.source == "rule_engine":
        return rule_result

    if not use_llm_fallback:
        return rule_result

    return generate_main_health_chatbot_llm_response(
        input_data,
        use_real_llm=use_real_llm,
    )


def with_fallback_metadata(rule_response, failed_rewrite_response):
    rule_response.safety_result = {
        **rule_response.safety_result,
        "fallback_used": True,
        "fallback_reason": "llm_rewrite_failed_safety_or_grounding",
        "failed_rewrite": {
            "source": failed_rewrite_response.source,
            "intent": failed_rewrite_response.intent,
            "is_safe": failed_rewrite_response.is_safe,
            "safety_result": failed_rewrite_response.safety_result,
        },
    }
    rule_response.is_safe = rule_response.safety_result["is_safe"]
    return rule_response
