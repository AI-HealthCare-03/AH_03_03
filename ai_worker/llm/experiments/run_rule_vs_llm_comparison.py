import argparse
from typing import NamedTuple

from ai_worker.llm.experiments.dummy_health_cases import (
    get_main_health_chatbot_dummy_cases,
    get_result_chatbot_dummy_cases,
)
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
from ai_worker.llm.schemas import MainHealthChatbotInput, ResultChatbotInput


class ComparisonCase(NamedTuple):
    case_id: str
    response_type: str
    input_data: ResultChatbotInput | MainHealthChatbotInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rule engine chatbot responses with LLM responses.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all comparison cases instead of the default three cases.",
    )
    parser.add_argument(
        "--use-real-llm",
        action="store_true",
        help="Call the real LLM API instead of using the local stub.",
    )
    parser.add_argument(
        "--confirm-all",
        action="store_true",
        help="Required when using --all and --use-real-llm together.",
    )
    return parser.parse_args()


def build_default_cases() -> list[ComparisonCase]:
    result_cases = get_result_chatbot_dummy_cases()
    main_cases = get_main_health_chatbot_dummy_cases()

    return [
        ComparisonCase(
            case_id="result_diabetes_001",
            response_type="result_chatbot",
            input_data=result_cases[0],
        ),
        ComparisonCase(
            case_id="result_hypertension_001",
            response_type="result_chatbot",
            input_data=result_cases[1],
        ),
        ComparisonCase(
            case_id="main_medication_safety_001",
            response_type="main_health_chatbot",
            input_data=main_cases[1],
        ),
    ]


def build_all_cases() -> list[ComparisonCase]:
    cases: list[ComparisonCase] = []

    for index, input_data in enumerate(get_result_chatbot_dummy_cases(), start=1):
        cases.append(
            ComparisonCase(
                case_id=f"result_case_{index:03d}",
                response_type="result_chatbot",
                input_data=input_data,
            )
        )

    for index, input_data in enumerate(get_main_health_chatbot_dummy_cases(), start=1):
        cases.append(
            ComparisonCase(
                case_id=f"main_case_{index:03d}",
                response_type="main_health_chatbot",
                input_data=input_data,
            )
        )

    return cases


def run_comparison_case(
    case: ComparisonCase,
    use_real_llm: bool,
) -> None:
    if case.response_type == "result_chatbot":
        input_data = case.input_data
        if not isinstance(input_data, ResultChatbotInput):
            raise TypeError("result_chatbot case requires ResultChatbotInput.")

        rule_response = generate_result_chatbot_response(input_data)
        llm_response = generate_result_chatbot_llm_response(
            input_data,
            use_real_llm=use_real_llm,
        )
        rewrite_response = rewrite_result_chatbot_response_with_llm(
            input_data,
            rule_engine_output=rule_response,
            use_real_llm=use_real_llm,
        )
    else:
        input_data = case.input_data
        if not isinstance(input_data, MainHealthChatbotInput):
            raise TypeError("main_health_chatbot case requires MainHealthChatbotInput.")

        rule_response = generate_main_health_chatbot_response(input_data)
        llm_response = generate_main_health_chatbot_llm_response(
            input_data,
            use_real_llm=use_real_llm,
        )
        rewrite_response = rewrite_main_health_chatbot_response_with_llm(
            input_data,
            rule_engine_output=rule_response,
            use_real_llm=use_real_llm,
        )

    print(f"\n--- Case: {case.case_id} ---")
    print(f"chatbot_type={case.response_type}")
    print(f"user_message={input_data.user_message}")

    print("\n[Rule Engine]")
    print(f"source={rule_response.source}")
    print(f"intent={rule_response.intent}")
    print(f"is_safe={rule_response.is_safe}")
    print(f"answer={rule_response.answer}")

    print("\n[LLM]")
    print(f"source={llm_response.source}")
    print(f"intent={llm_response.intent}")
    print(f"is_safe={llm_response.is_safe}")
    print(f"answer={llm_response.answer}")
    print(f"safety_result={llm_response.safety_result}")
    print(f"grounding_result={llm_response.safety_result.get('grounding_result')}")

    print("\n[LLM Rewrite]")
    print(f"source={rewrite_response.source}")
    print(f"intent={rewrite_response.intent}")
    print(f"is_safe={rewrite_response.is_safe}")
    print(f"answer={rewrite_response.answer}")
    print(f"safety_result={rewrite_response.safety_result}")
    print(f"grounding_result={rewrite_response.safety_result.get('grounding_result')}")


def main() -> None:
    args = parse_args()

    if args.all and args.use_real_llm and not args.confirm_all:
        raise SystemExit(
            "--all and --use-real-llm together can trigger multiple API calls. "
            "Re-run with --confirm-all if you intentionally want this."
        )

    cases = build_all_cases() if args.all else build_default_cases()

    print("\n=== Rule Engine vs LLM Comparison ===")
    print(f"case_count={len(cases)}")
    print(f"use_real_llm={args.use_real_llm}")

    for case in cases:
        run_comparison_case(
            case,
            use_real_llm=args.use_real_llm,
        )


if __name__ == "__main__":
    main()
