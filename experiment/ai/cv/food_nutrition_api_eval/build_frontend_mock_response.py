from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NUTRITION_KEYS = ["energy_kcal", "carbohydrate_g", "protein_g", "fat_g", "sodium_mg"]
CONFIRMATION_MESSAGE = "정확한 영양성분 계산을 위해 가장 가까운 음식을 선택해주세요."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build experiment-only frontend mock diet analysis responses from service response samples."
    )
    parser.add_argument("--service-response", required=True, help="service_response_samples.json path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    service_response = json.loads(Path(args.service_response).read_text(encoding="utf-8"))
    foods = list(service_response.get("foods", []))
    full_response = build_mock_response(foods, analysis_id="mock-diet-analysis-001")
    auto_response = build_mock_response(
        [food for food in foods if is_auto_confirmed(food)],
        analysis_id="mock-diet-analysis-auto-confirmed",
    )
    needs_response = build_mock_response(
        [food for food in foods if is_confirmation_food(food)],
        analysis_id="mock-diet-analysis-needs-confirmation",
    )
    no_candidates_response = build_mock_response(
        [food for food in foods if is_no_candidate_food(food)],
        analysis_id="mock-diet-analysis-no-candidates",
    )

    write_json(output_dir / "diet_analysis_mock_full.json", full_response)
    write_json(output_dir / "diet_analysis_mock_auto_confirmed.json", auto_response)
    write_json(output_dir / "diet_analysis_mock_needs_confirmation.json", needs_response)
    write_json(output_dir / "diet_analysis_mock_no_candidates.json", no_candidates_response)
    (output_dir / "frontend_mock_report.md").write_text(
        build_report(full_response, auto_response, needs_response, no_candidates_response),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'diet_analysis_mock_full.json'}")
    print(f"Wrote {output_dir / 'diet_analysis_mock_auto_confirmed.json'}")
    print(f"Wrote {output_dir / 'diet_analysis_mock_needs_confirmation.json'}")
    print(f"Wrote {output_dir / 'diet_analysis_mock_no_candidates.json'}")
    print(f"Wrote {output_dir / 'frontend_mock_report.md'}")


def build_mock_response(foods: list[dict[str, Any]], *, analysis_id: str) -> dict[str, Any]:
    auto_confirmed_foods = [
        build_auto_confirmed_food(food, idx) for idx, food in enumerate(foods, start=1) if is_auto_confirmed(food)
    ]
    needs_confirmation_foods = [
        build_needs_confirmation_food(food, idx)
        for idx, food in enumerate(foods, start=1)
        if is_confirmation_food(food)
    ]
    no_candidate_foods = [
        build_no_candidate_food(food, idx) for idx, food in enumerate(foods, start=1) if is_no_candidate_food(food)
    ]
    summary = build_summary(
        detected_food_count=len(foods),
        auto_confirmed_foods=auto_confirmed_foods,
        needs_confirmation_foods=needs_confirmation_foods,
        no_candidate_foods=no_candidate_foods,
    )
    return {
        "analysis_id": analysis_id,
        "status": "needs_user_confirmation"
        if summary["needs_user_confirmation_count"] > 0 or summary["no_candidates_count"] > 0
        else "completed",
        "summary": summary,
        "auto_confirmed_foods": auto_confirmed_foods,
        "needs_confirmation_foods": needs_confirmation_foods,
        "no_candidate_foods": no_candidate_foods,
        "messages": build_messages(summary),
    }


def build_summary(
    *,
    detected_food_count: int,
    auto_confirmed_foods: list[dict[str, Any]],
    needs_confirmation_foods: list[dict[str, Any]],
    no_candidate_foods: list[dict[str, Any]],
) -> dict[str, Any]:
    totals = {f"total_{key}": 0.0 for key in NUTRITION_KEYS}
    for food in auto_confirmed_foods:
        nutrition = food.get("nutrition", {})
        for key in NUTRITION_KEYS:
            totals[f"total_{key}"] += float(nutrition.get(key) or 0.0)
    totals = {key: round(value, 4) for key, value in totals.items()}
    pending_count = len(needs_confirmation_foods) + len(no_candidate_foods)
    return {
        "detected_food_count": detected_food_count,
        "auto_confirmed_count": len(auto_confirmed_foods),
        "needs_user_confirmation_count": len(needs_confirmation_foods) + len(no_candidate_foods),
        "no_candidates_count": len(no_candidate_foods),
        **totals,
        "nutrition_calculation_status": "partial" if pending_count else "completed",
    }


def build_auto_confirmed_food(food: dict[str, Any], index: int) -> dict[str, Any]:
    selected = food.get("selected_candidate") if isinstance(food.get("selected_candidate"), dict) else {}
    nutrition = selected.get("nutrition") if isinstance(selected.get("nutrition"), dict) else {}
    return {
        "food_item_id": f"food-{index:03d}",
        "vision_food_name": food.get("vision_food_name") or "",
        "display_name": selected.get("food_name") or food.get("vision_food_name") or "",
        "source": selected.get("source") or "",
        "food_code": selected.get("food_code") or "",
        "serving_size": selected.get("serving_size") or None,
        "nutrition": {key: optional_float(nutrition.get(key)) for key in NUTRITION_KEYS},
        "editable": True,
        "user_action": "can_edit",
    }


def build_needs_confirmation_food(food: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "food_item_id": f"food-{index:03d}",
        "vision_food_name": food.get("vision_food_name") or "",
        "raw_food_name": food.get("raw_food_name") or "",
        "nutrition_status": food.get("nutrition_status") or "",
        "match_status": food.get("match_status") or "",
        "candidates": [
            build_candidate(candidate, index, candidate_index)
            for candidate_index, candidate in enumerate(food.get("candidates", [])[:3], start=1)
        ],
        "message": CONFIRMATION_MESSAGE,
        "editable": True,
        "user_action": "select_candidate",
    }


def build_candidate(candidate: dict[str, Any], food_index: int, candidate_index: int) -> dict[str, Any]:
    return {
        "candidate_id": f"food-{food_index:03d}-candidate-{candidate_index:02d}",
        "source": candidate.get("source") or "",
        "food_name": candidate.get("food_name") or "",
        "food_code": candidate.get("food_code") or "",
        "match_status": candidate.get("match_status") or "",
        "rank_score": optional_float(candidate.get("rank_score")),
        "rank_reason": candidate.get("rank_reason") or "",
        "serving_size": candidate.get("serving_size") or None,
        "nutrition_preview": {key: optional_float(candidate.get(key)) for key in NUTRITION_KEYS},
    }


def build_no_candidate_food(food: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "food_item_id": f"food-{index:03d}",
        "vision_food_name": food.get("vision_food_name") or "",
        "raw_food_name": food.get("raw_food_name") or "",
        "nutrition_status": food.get("nutrition_status") or "",
        "message": "영양성분 후보를 찾지 못했습니다. 음식을 직접 입력하거나 다시 검색해주세요.",
        "editable": True,
        "user_action": "manual_search_required",
    }


def build_messages(summary: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    if summary["needs_user_confirmation_count"] > 0:
        messages.append("일부 음식은 영양성분 계산 전 사용자 확인이 필요합니다.")
    if summary["no_candidates_count"] > 0:
        messages.append("일부 음식은 후보를 찾지 못했습니다. 직접 입력 또는 재검색이 필요합니다.")
    if summary["nutrition_calculation_status"] == "partial":
        messages.append("현재 총 영양성분은 자동 확정된 음식만 합산한 값입니다.")
    return messages


def build_report(
    full_response: dict[str, Any],
    auto_response: dict[str, Any],
    needs_response: dict[str, Any],
    no_candidates_response: dict[str, Any],
) -> str:
    lines = [
        "# Frontend Diet Analysis Mock Report",
        "",
        "## 전체 요약",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in full_response["summary"].items())
    append_sample_section(lines, "자동 확정 음식 샘플", auto_response["auto_confirmed_foods"])
    append_sample_section(lines, "사용자 확인 필요 음식 샘플", needs_response["needs_confirmation_foods"])
    append_sample_section(lines, "후보 없음 음식 샘플", no_candidates_response["no_candidate_foods"])
    lines.extend(
        [
            "",
            "## 프론트 UI 표시 정책",
            "",
            "- `auto_confirmed_foods`: 결과 카드에 바로 표시하고 수량/음식명을 수정할 수 있게 둡니다.",
            "- `needs_confirmation_foods`: 후보 선택 리스트를 보여주고 선택 전에는 총 영양성분에 합산하지 않습니다.",
            "- `no_candidate_foods`: 직접 검색/직접 입력 플로우로 보냅니다.",
            "- `summary.nutrition_calculation_status=partial`이면 총합이 임시값임을 표시합니다.",
            "",
            "## Production API 전환 시 필요한 필드",
            "",
            "- 분석 식별자: `analysis_id`",
            "- 음식 item 식별자: `food_item_id`",
            "- 사용자 선택 candidate 저장용 `candidate_id`, `food_code`",
            "- serving/portion 보정 필드",
            "- 사용자 확정 상태: `user_action`, `auto_confirmed`, `needs_user_confirmation`",
            "- 부분 합산 여부: `nutrition_calculation_status`",
        ]
    )
    return "\n".join(lines) + "\n"


def append_sample_section(lines: list[str], title: str, foods: list[dict[str, Any]]) -> None:
    lines.extend(["", f"## {title}", ""])
    if not foods:
        lines.append("- none")
        return
    for food in foods[:10]:
        lines.append(
            "- {food_item_id}: {name} ({action})".format(
                food_item_id=food.get("food_item_id"),
                name=food.get("display_name") or food.get("vision_food_name") or food.get("raw_food_name"),
                action=food.get("user_action"),
            )
        )


def is_auto_confirmed(food: dict[str, Any]) -> bool:
    return bool(food.get("auto_confirmed")) and food.get("match_status") == "matched"


def is_no_candidate_food(food: dict[str, Any]) -> bool:
    return str(food.get("nutrition_status") or "") in {"no_candidates", "no_query"}


def is_confirmation_food(food: dict[str, Any]) -> bool:
    return bool(food.get("needs_user_confirmation")) and not is_no_candidate_food(food)


def optional_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
