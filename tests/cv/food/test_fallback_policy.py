from ai_runtime.cv.food.fallback_policy import select_food_detection_candidate
from ai_runtime.cv.food.schemas import FoodDetectionCandidateSet


def test_default_policy_uses_rule_based_and_does_not_call_gpt_vision() -> None:
    candidate = select_food_detection_candidate(
        cv_result=None,
        rule_based_foods=[
            {"name": "현미밥", "confidence": 0.9},
            {"name": "닭가슴살", "confidence": 0.8},
        ],
    )

    assert candidate.provider == "rule_based_food_detection"
    assert candidate.detected_foods == ["현미밥", "닭가슴살"]
    assert candidate.confidence == 0.85
    assert candidate.needs_review is False
    assert candidate.metadata["gpt_vision_called"] is False
    assert candidate.should_use_nutrition_scorer()
    assert candidate.to_scorer_foods()[0]["name"] == "현미밥"


def test_low_confidence_cv_result_is_marked_needs_review_without_gpt_call() -> None:
    candidate = select_food_detection_candidate(
        cv_result=FoodDetectionCandidateSet(
            provider="cv_model",
            detected_foods=["샐러드"],
            confidence=0.42,
        ),
        gpt_vision_fallback_enabled=False,
    )

    assert candidate.provider == "cv_model"
    assert candidate.needs_review is True
    assert candidate.fallback_reason == "cv_confidence_below_threshold"
    assert not candidate.should_use_nutrition_scorer()


def test_gpt_vision_fallback_is_only_candidate_when_flag_enabled() -> None:
    candidate = select_food_detection_candidate(
        cv_result=FoodDetectionCandidateSet(
            provider="cv_model",
            detected_foods=["소스가 많은 음식"],
            confidence=0.3,
        ),
        gpt_vision_fallback_enabled=True,
    )

    assert candidate.provider == "gpt_vision"
    assert candidate.detected_foods == []
    assert candidate.needs_review is True
    assert candidate.fallback_reason == "gpt_vision_candidate_requires_user_confirmation"
    assert candidate.metadata["called"] is False
    assert candidate.raw_output["fallback_candidate"] is True
