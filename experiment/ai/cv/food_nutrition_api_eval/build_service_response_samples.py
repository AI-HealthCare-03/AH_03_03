from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

AUTO_CONFIRM_MATCH_STATUS = "matched"
LOOKUP_SUCCESS_STATUSES = {"matched", "likely_match", "multiple_candidates", "weak_match", "fallback_used"}
CONFIRMATION_STATUSES = {
    "weak_match",
    "multiple_candidates",
    "fallback_used",
    "likely_match",
    "no_candidates",
    "no_query",
    "api_unavailable",
    "parse_failed",
}
NUTRITION_FIELDS = ["energy_kcal", "carbohydrate_g", "protein_g", "fat_g", "sodium_mg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build experiment-only service response samples from GPT Vision and MFDS nutrition outputs."
    )
    parser.add_argument("--vision-predictions", required=True, help="GPT Vision predictions.csv path.")
    parser.add_argument("--nutrition-predictions", required=True, help="Nutrition predictions.csv path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_rows = load_csv(Path(args.vision_predictions))
    nutrition_rows = load_csv(Path(args.nutrition_predictions))
    response = build_response(vision_rows, nutrition_rows)

    json_path = output_dir / "service_response_samples.json"
    report_path = output_dir / "service_response_report.md"
    json_path.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_report(response, vision_rows), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{key: str(value or "") for key, value in row.items()} for row in csv.DictReader(f)]


def build_response(vision_rows: list[dict[str, str]], nutrition_rows: list[dict[str, str]]) -> dict[str, Any]:
    foods = [build_food_item(row) for row in nutrition_rows]
    auto_confirmed_count = sum(bool(item["auto_confirmed"]) for item in foods)
    needs_user_confirmation_count = sum(bool(item["needs_user_confirmation"]) for item in foods)
    no_query_count = sum(item["nutrition_status"] == "no_query" for item in foods)
    lookup_success_count = sum(item["nutrition_status"] in LOOKUP_SUCCESS_STATUSES for item in foods)
    summary = {
        "total_image_rows": len(vision_rows),
        "detected_food_count": len(foods),
        "auto_confirmed_count": auto_confirmed_count,
        "needs_user_confirmation_count": needs_user_confirmation_count,
        "no_query_count": no_query_count,
        "no_candidates_count": sum(item["nutrition_status"] == "no_candidates" for item in foods),
        "nutrition_lookup_success_rate": rate(lookup_success_count, len(foods)),
    }
    return {"summary": summary, "foods": foods}


def build_food_item(row: dict[str, str]) -> dict[str, Any]:
    match_status = row.get("match_status") or row.get("status") or ""
    status = row.get("status") or match_status
    auto_confirmed = match_status == AUTO_CONFIRM_MATCH_STATUS
    needs_user_confirmation = not auto_confirmed or parse_bool(row.get("needs_user_confirmation"))
    item: dict[str, Any] = {
        "row_id": parse_int(row.get("row_id")),
        "vision_food_name": row.get("query") or row.get("original_query") or "",
        "raw_food_name": row.get("query") or "",
        "nutrition_status": status,
        "match_status": match_status,
        "auto_confirmed": auto_confirmed,
        "needs_user_confirmation": needs_user_confirmation,
        "selected_candidate": build_selected_candidate(row) if auto_confirmed else None,
        "candidates": [] if auto_confirmed else build_candidates(row),
    }
    if not auto_confirmed:
        if status in {"no_candidates", "no_query"}:
            item["message"] = "영양성분 후보를 찾지 못했습니다. 음식을 직접 입력하거나 다시 선택해주세요."
        elif status in {"api_unavailable", "parse_failed"}:
            item["message"] = "영양성분 조회가 일시적으로 실패했습니다. 잠시 후 다시 시도해주세요."
        else:
            item["message"] = "정확한 영양성분 계산을 위해 가장 가까운 음식을 선택해주세요."
    return item


def build_selected_candidate(row: dict[str, str]) -> dict[str, Any]:
    return {
        "source": row.get("source") or row.get("provider") or "",
        "food_name": row.get("matched_food_name") or "",
        "food_code": row.get("matched_food_code") or "",
        "serving_size": row.get("serving_size") or None,
        "nutrition": {field: parse_float(row.get(field)) for field in NUTRITION_FIELDS},
    }


def build_candidates(row: dict[str, str]) -> list[dict[str, Any]]:
    matched_name = row.get("matched_food_name") or ""
    if not matched_name:
        return []
    return [
        {
            "source": row.get("source") or row.get("provider") or "",
            "food_name": matched_name,
            "food_code": row.get("matched_food_code") or None,
            "match_status": row.get("match_status") or row.get("status") or "",
            "rank_score": parse_float(row.get("rank_score")),
            "rank_reason": row.get("rank_reason") or "",
        }
    ]


def build_report(response: dict[str, Any], vision_rows: list[dict[str, str]]) -> str:
    foods = response["foods"]
    auto_foods = [item for item in foods if item["auto_confirmed"]]
    confirmation_foods = [item for item in foods if item["needs_user_confirmation"]]
    no_query_foods = [item for item in foods if item["nutrition_status"] == "no_query"]
    no_candidate_foods = [item for item in foods if item["nutrition_status"] == "no_candidates"]
    risky_foods = [
        item
        for item in foods
        if item["nutrition_status"] in {"weak_match", "multiple_candidates", "fallback_used", "likely_match"}
    ]
    lines = [
        "# Service Response Sample Report",
        "",
        "## 전체 요약",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in response["summary"].items())
    append_food_list(lines, "자동 확정 음식 목록", auto_foods)
    append_food_list(lines, "사용자 확인 필요 음식 목록", confirmation_foods)
    append_food_list(lines, "no_query 목록", no_query_foods)
    append_food_list(lines, "no_candidates 목록", no_candidate_foods)
    append_food_list(lines, "위험 케이스 요약", risky_foods)
    append_vision_mismatch_section(lines, vision_rows)
    lines.extend(
        [
            "",
            "## 프론트 표시 정책 제안",
            "",
            "- `auto_confirmed=true`인 음식은 영양성분을 바로 표시하되 사용자가 수정할 수 있게 둡니다.",
            "- `needs_user_confirmation=true`인 음식은 후보 선택 UI를 먼저 보여주고 확정 전에는 영양성분 합산에 넣지 않습니다.",
            "- `no_candidates` 또는 `no_query`는 직접 입력/검색 UI로 보냅니다.",
            "- `api_unavailable`은 재시도 버튼과 임시 안내 문구를 보여줍니다.",
            "- production 응답에는 평가용 `expected_foods`를 포함하지 않습니다.",
            "",
            "## Production API 전환 시 필요한 필드",
            "",
            "- `row_id` 또는 image-local food id",
            "- `vision_food_name`, `raw_food_name`",
            "- `nutrition_status`, `match_status`",
            "- `auto_confirmed`, `needs_user_confirmation`",
            "- `selected_candidate`",
            "- `candidates`",
            "- user-selected candidate id/code",
            "- serving amount and user adjustment fields",
        ]
    )
    return "\n".join(lines) + "\n"


def append_food_list(lines: list[str], title: str, foods: list[dict[str, Any]]) -> None:
    lines.extend(["", f"## {title}", ""])
    if not foods:
        lines.append("- none")
        return
    for item in foods:
        candidate = item.get("selected_candidate") or {}
        candidates = item.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
        candidate_name = (candidate.get("food_name") if isinstance(candidate, dict) else "") or ""
        lines.append(
            "- row_id={row_id}, vision={vision}, status={status}, candidate={candidate}".format(
                row_id=item.get("row_id"),
                vision=item.get("vision_food_name") or "",
                status=item.get("nutrition_status") or "",
                candidate=candidate_name,
            )
        )


def append_vision_mismatch_section(lines: list[str], vision_rows: list[dict[str, str]]) -> None:
    lines.extend(["", "## Debug/Eval: GPT Vision label mismatch", ""])
    mismatches: list[str] = []
    for row in vision_rows:
        expected = split_names(row.get("expected_foods", ""))
        raw_names = split_names(row.get("raw_food_names", ""))
        if expected and raw_names and not set(expected).intersection(raw_names):
            mismatches.append(
                "- row_id={row_id}, expected={expected}, vision={vision}".format(
                    row_id=row.get("row_id") or "",
                    expected="|".join(expected),
                    vision="|".join(raw_names),
                )
            )
    if mismatches:
        lines.extend(mismatches)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("`expected_foods`는 실험 평가용이며 production 응답 JSON에는 포함하지 않았습니다.")


def split_names(value: str) -> list[str]:
    return [item.strip() for item in value.replace("|", ",").split(",") if item.strip()]


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def parse_int(value: str | None) -> int | str:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return str(value or "")


def parse_float(value: str | None) -> float | None:
    try:
        text = str(value or "").strip()
        return float(text) if text else None
    except ValueError:
        return None


def rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


if __name__ == "__main__":
    main()
