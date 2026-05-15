from ai_worker.llm.experiments.dummy_health_cases import (
    get_main_health_chatbot_dummy_cases,
    get_result_chatbot_dummy_cases,
)
from ai_worker.llm.health_chatbot import (
    generate_main_health_chatbot_response,
    generate_result_chatbot_response,
)
from ai_worker.llm.recommendation_message import generate_recommendation_message
from ai_worker.llm.schemas import RecommendationMessageInput


def run_recommendation_message_case() -> None:
    sample_case = get_result_chatbot_dummy_cases()[0]
    result = generate_recommendation_message(
        RecommendationMessageInput(
            risk_factors=sample_case.risk_factors,
            recommended_challenges=sample_case.recommended_challenges,
            tone=sample_case.tone,
        )
    )

    print("\n=== Recommendation Message Result ===")
    print(result.model_dump_json(indent=2))


def run_result_chatbot_cases() -> None:
    print("\n=== Result Chatbot Results ===")
    for index, case in enumerate(get_result_chatbot_dummy_cases(), start=1):
        result = generate_result_chatbot_response(case)
        print(f"\n--- Result Case {index}: {case.user_message} ---")
        print(result.model_dump_json(indent=2))


def run_main_health_chatbot_cases() -> None:
    print("\n=== Main Health Chatbot Results ===")
    for index, case in enumerate(get_main_health_chatbot_dummy_cases(), start=1):
        result = generate_main_health_chatbot_response(case)
        print(f"\n--- Main Case {index}: {case.user_message} ---")
        print(result.model_dump_json(indent=2))


def main() -> None:
    run_recommendation_message_case()
    run_result_chatbot_cases()
    run_main_health_chatbot_cases()


if __name__ == "__main__":
    main()
