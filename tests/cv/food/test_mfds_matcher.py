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
    assert result.metadata["top_candidates"] == [
        {
            "food_name": "비빔국수",
            "food_code": "D303-157000000-0001",
            "rank_score": result.metadata["rank_score"],
            "rank_reason": result.metadata["rank_reason"],
            "nutrition": None,
        }
    ]
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


def test_mfds_matcher_retries_transient_request_failure() -> None:
    calls: list[str] = []

    def flaky_fetch_payload(query: str) -> dict:
        calls.append(query)
        if len(calls) == 1:
            raise RuntimeError("temporary timeout")
        return {
            "response": {
                "body": {
                    "items": {
                        "item": [
                            {
                                "FOOD_NM_KR": "핫초코",
                                "FOOD_CD": "D701-001",
                            }
                        ]
                    }
                }
            }
        }

    matcher = MfdsFoodDbMatcher(
        service_key="test-key",
        fetch_payload=flaky_fetch_payload,
        retry_backoff_seconds=(0,),
    )

    result = matcher.match("핫초코")

    assert calls == ["핫초코", "핫초코"]
    assert result.matched_food_name == "핫초코"
    assert result.match_source == "mfds_matched"


def test_mfds_matcher_keeps_fallback_when_retries_fail() -> None:
    calls = 0

    def failing_fetch_payload(_: str) -> dict:
        nonlocal calls
        calls += 1
        raise RuntimeError("timeout")

    matcher = MfdsFoodDbMatcher(
        service_key="test-key",
        fallback_matcher=LocalFallbackFoodDbMatcher(),
        fetch_payload=failing_fetch_payload,
        retry_backoff_seconds=(0,),
    )

    result = matcher.match("쌀떡")

    assert calls == 3
    assert result.matched_food_name == "가래떡"
    assert result.match_source == "local_stub_alias"


def test_mfds_matcher_uses_experiment_fallback_query_rules() -> None:
    calls: list[str] = []

    def fake_fetch_payload(query: str) -> dict:
        calls.append(query)
        if query != "냉면":
            return {"response": {"body": {"items": []}}}
        return {
            "response": {
                "body": {
                    "items": {
                        "item": [
                            {
                                "FOOD_NM_KR": "비빔냉면",
                                "FOOD_CD": "D302-001",
                            }
                        ]
                    }
                }
            }
        }

    matcher = MfdsFoodDbMatcher(service_key="test-key", fetch_payload=fake_fetch_payload)

    result = matcher.match("비빔냉면")

    assert calls == ["비빔냉면", "냉면"]
    assert result.matched_food_name == "비빔냉면"
    assert result.metadata is not None
    assert result.metadata["used_query"] == "냉면"


def test_mfds_matcher_stores_top_candidates_without_raw_payload() -> None:
    def fake_fetch_payload(_: str) -> dict:
        return {
            "response": {
                "body": {
                    "items": {
                        "item": [
                            {"FOOD_NM_KR": "짜장면", "FOOD_CD": "D403-001", "SECRET_RAW": "do-not-store"},
                            {"FOOD_NM_KR": "짜장면 곱빼기", "FOOD_CD": "D403-002", "SECRET_RAW": "do-not-store"},
                            {"FOOD_NM_KR": "짜장면 소스", "FOOD_CD": "D403-003", "SECRET_RAW": "do-not-store"},
                            {"FOOD_NM_KR": "짜장라면", "FOOD_CD": "D403-004", "SECRET_RAW": "do-not-store"},
                            {"FOOD_NM_KR": "중식 짜장면", "FOOD_CD": "D403-005", "SECRET_RAW": "do-not-store"},
                            {"FOOD_NM_KR": "간짜장", "FOOD_CD": "D403-006", "SECRET_RAW": "do-not-store"},
                        ]
                    }
                }
            }
        }

    matcher = MfdsFoodDbMatcher(service_key="test-key", fetch_payload=fake_fetch_payload, max_candidates=6)

    result = matcher.match("짜장면")

    assert result.metadata is not None
    top_candidates = result.metadata["top_candidates"]
    assert isinstance(top_candidates, list)
    assert len(top_candidates) == 5
    assert top_candidates[0]["food_name"] == "짜장면"
    assert set(top_candidates[0]) == {"food_name", "food_code", "rank_score", "rank_reason", "nutrition"}
    assert "SECRET_RAW" not in str(top_candidates)


def test_mfds_matcher_normalizes_nutrition_from_fixture_payload() -> None:
    def fake_fetch_payload(_: str) -> dict:
        return {
            "response": {
                "body": {
                    "items": {
                        "item": [
                            {
                                "FOOD_NM_KR": "비빔밥",
                                "FOOD_CD": "D101-001",
                                "SERVING_SIZE": "100g",
                                "AMT_NUM1": "650",
                                "AMT_NUM6": "78.2",
                                "AMT_NUM3": "25",
                                "AMT_NUM4": "22.5",
                                "AMT_NUM13": "900",
                                "SECRET_RAW": "do-not-store",
                                "serviceKey": "do-not-store",
                            }
                        ]
                    }
                }
            }
        }

    matcher = MfdsFoodDbMatcher(service_key="test-key", fetch_payload=fake_fetch_payload)

    result = matcher.match("비빔밥")

    assert result.metadata is not None
    assert result.metadata["nutrition"] == {
        "calories_kcal": 650.0,
        "carbohydrate_g": 78.2,
        "protein_g": 25.0,
        "fat_g": 22.5,
        "sodium_mg": 900.0,
        "basis_amount": 100.0,
        "basis_unit": "g",
        "basis_label": "100g 기준",
        "serving_size": "100g",
        "serving_reference": "100g",
    }
    top_candidate = result.metadata["top_candidates"][0]
    assert top_candidate["nutrition"] == result.metadata["nutrition"]
    assert "SECRET_RAW" not in str(result.metadata)
    assert "serviceKey" not in str(result.metadata)


def test_mfds_matcher_marks_nutrition_basis_unknown_when_serving_size_missing() -> None:
    def fake_fetch_payload(_: str) -> dict:
        return {
            "response": {
                "body": {
                    "items": {
                        "item": [
                            {
                                "FOOD_NM_KR": "짜장면",
                                "FOOD_CD": "D403-001",
                                "AMT_NUM1": "720",
                                "AMT_NUM6": "95",
                            }
                        ]
                    }
                }
            }
        }

    matcher = MfdsFoodDbMatcher(service_key="test-key", fetch_payload=fake_fetch_payload)

    result = matcher.match("짜장면")

    assert result.metadata is not None
    nutrition = result.metadata["nutrition"]
    assert nutrition["calories_kcal"] == 720.0
    assert nutrition["carbohydrate_g"] == 95.0
    assert nutrition["basis_amount"] is None
    assert nutrition["basis_unit"] is None
    assert nutrition["basis_label"] == "기준량 확인 필요"


def test_mfds_matcher_preserves_serving_reference_and_food_weight_metadata() -> None:
    def fake_fetch_payload(_: str) -> dict:
        return {
            "response": {
                "body": {
                    "items": {
                        "item": [
                            {
                                "FOOD_NM_KR": "돌솥비빔밥",
                                "FOOD_CD": "D101-002",
                                "영양성분 함량 기준량": "1회 제공량",
                                "1회섭취 참고량": "350 g",
                                "식품중량": "420 g",
                                "AMT_NUM1": "820",
                                "AMT_NUM3": "28",
                                "AMT_NUM4": "24",
                                "AMT_NUM6": "112",
                                "AMT_NUM13": "1150",
                                "raw_url": "do-not-store",
                            }
                        ]
                    }
                }
            }
        }

    matcher = MfdsFoodDbMatcher(service_key="test-key", fetch_payload=fake_fetch_payload)

    result = matcher.match("돌솥비빔밥")

    assert result.metadata is not None
    nutrition = result.metadata["nutrition"]
    assert nutrition["calories_kcal"] == 820.0
    assert nutrition["basis_amount"] == 1
    assert nutrition["basis_unit"] == "serving"
    assert nutrition["basis_label"] == "1회 제공량 기준"
    assert nutrition["basis_reference"] == "1회 제공량"
    assert nutrition["serving_reference"] == "350 g"
    assert nutrition["food_weight"] == "420 g"
    assert "raw_url" not in str(result.metadata)
