from __future__ import annotations

from ai_runtime.cv.food.matcher import LocalFallbackFoodDbMatcher
from ai_runtime.cv.food.nutrition.providers.mfds import MfdsFoodDbMatcher


def test_mfds_matcher_enriches_food_match_from_fixture_payload() -> None:
    def fake_fetch_payload(query: str) -> dict:
        assert query == "비빔국수"
        return {
            "response": {
                "body": {
                    "items": {
                        "item": [
                            {
                                "FOOD_NM_KR": "비빔국수",
                                "FOOD_CD": "D303-157000000-0001",
                            }
                        ]
                    }
                }
            }
        }

    matcher = MfdsFoodDbMatcher(service_key="test-key", fetch_payload=fake_fetch_payload)
    result = matcher.match("비빔국수")

    assert result.original_name == "비빔국수"
    assert result.query_name == "비빔국수"
    assert result.matched_food_name == "비빔국수"
    assert result.matched_food_code == "D303-157000000-0001"
    assert result.match_source == "mfds_matched"
    assert result.match_confidence == 1.0
    assert result.needs_user_confirmation is True
    assert result.metadata is not None
    assert result.metadata["provider"] == "mfds"
    assert result.metadata["status"] == "matched"
    assert result.metadata["candidate_count"] == 1
    assert result.metadata["used_query"] == "비빔국수"
    assert result.metadata["rank_score"] >= 100
    assert "exact_normalized_match" in str(result.metadata["rank_reason"])
    assert isinstance(result.metadata["latency_ms"], int)


def test_mfds_matcher_marks_no_candidates_without_failing() -> None:
    matcher = MfdsFoodDbMatcher(service_key="test-key", fetch_payload=lambda _: {"response": {"body": {"items": []}}})

    result = matcher.match("생선찌개")

    assert result.matched_food_name is None
    assert result.matched_food_code is None
    assert result.match_source == "mfds_no_candidates"
    assert result.needs_user_confirmation is True


def test_mfds_matcher_uses_local_fallback_when_api_fails() -> None:
    def failing_fetch_payload(_: str) -> dict:
        raise RuntimeError("timeout")

    matcher = MfdsFoodDbMatcher(
        service_key="test-key",
        fallback_matcher=LocalFallbackFoodDbMatcher(),
        fetch_payload=failing_fetch_payload,
    )

    result = matcher.match("쌀떡")

    assert result.matched_food_name == "가래떡"
    assert result.matched_food_code == "local_stub:garaetteok"
    assert result.match_source == "local_stub_alias"
