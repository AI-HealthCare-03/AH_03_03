from ai_runtime.cv.food.matcher import match_food_name
from ai_runtime.cv.food.normalization import cleanup_food_query, normalize_food_name


def test_cleanup_food_query_keeps_search_query_without_over_canonicalizing() -> None:
    assert cleanup_food_query(" Rice Cake (steamed) ") == "rice cake"
    assert cleanup_food_query("가래떡(구운 것)") == "가래떡"
    assert normalize_food_name("rice cake") == "ricecake"


def test_local_matcher_matches_clear_rice_cake_aliases_to_candidate_garaetteok() -> None:
    for query in ["rice cake", "쌀떡", "garaetteok"]:
        match = match_food_name(query)

        assert match.query_name
        assert match.matched_food_name == "가래떡"
        assert match.matched_food_code == "local_stub:garaetteok"
        assert match.match_source == "local_stub_alias"
        assert match.needs_user_confirmation is False


def test_local_matcher_does_not_fix_generic_tteok_to_garaetteok() -> None:
    match = match_food_name("떡")

    assert match.original_name == "떡"
    assert match.query_name == "떡"
    assert match.matched_food_name is None
    assert match.matched_food_code is None
    assert match.match_source == "local_stub_unmatched"
    assert match.match_confidence is None
    assert match.needs_user_confirmation is True
