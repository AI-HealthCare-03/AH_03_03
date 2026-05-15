from ai_worker.llm.health_chatbot import (
    generate_main_health_chatbot_response,
    generate_result_chatbot_response,
)
from ai_worker.llm.llm_generator import (
    generate_main_health_chatbot_llm_response,
    generate_result_chatbot_llm_response,
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
    use_real_llm: bool = False,
) -> ResultChatbotOutput:
    rule_result = generate_result_chatbot_response(input_data)
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
    use_real_llm: bool = False,
) -> MainHealthChatbotOutput:
    rule_result = generate_main_health_chatbot_response(input_data)
    if rule_result.source == "rule_engine":
        return rule_result

    if not use_llm_fallback:
        return rule_result

    return generate_main_health_chatbot_llm_response(
        input_data,
        use_real_llm=use_real_llm,
    )
