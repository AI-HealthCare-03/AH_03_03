# Diet Image Analysis API Contract Draft

This document is an experiment-only production API contract draft. It does not change runtime code.

## 1. Feature Overview

Diet image analysis receives a meal image, extracts visible food candidates with GPT Vision, looks up nutrition candidates through MFDS, and separates foods into auto-confirmed and user-confirmation groups.

Current experiment reference:

- GPT Vision empty result count improved from `22/30` to `0/30`.
- MFDS nutrition lookup success rate: `0.8936`.
- Service response sample:
  - `detected_food_count`: 47
  - `auto_confirmed_count`: 8
  - `needs_user_confirmation_count`: 39
  - `no_candidates_count`: 4
  - `nutrition_calculation_status`: `partial`

Core policy:

- Only `matched` food items are auto-confirmed.
- `weak_match`, `multiple_candidates`, `fallback_used`, `no_candidates`, and `no_query` require user action.
- Total nutrition is calculated from confirmed foods only.
- Evaluation labels such as `expected_foods` must not be used in production.

## 2. Endpoint Draft

### POST `/api/v1/diets/analyze-image`

Starts image analysis.

Request:

- Content type: `multipart/form-data`
- Fields:
  - `image`: required image file
  - `meal_type`: optional, for example `breakfast`, `lunch`, `dinner`, `snack`
  - `taken_at`: optional ISO 8601 timestamp

Response `202 Accepted`:

```json
{
  "analysis_id": "diet-analysis-20260610-0001",
  "status": "processing",
  "message": "식단 이미지 분석을 시작했습니다."
}
```

### GET `/api/v1/diets/analyses/{analysis_id}`

Returns the current analysis result. The response shape follows the frontend mock fixture.

Response `200 OK`:

```json
{
  "analysis_id": "diet-analysis-20260610-0001",
  "status": "needs_user_confirmation",
  "summary": {
    "detected_food_count": 47,
    "auto_confirmed_count": 8,
    "needs_user_confirmation_count": 39,
    "no_candidates_count": 4,
    "total_energy_kcal": 831.0,
    "total_carbohydrate_g": 105.83,
    "total_protein_g": 44.74,
    "total_fat_g": 26.68,
    "total_sodium_mg": 4021.0,
    "nutrition_calculation_status": "partial"
  },
  "auto_confirmed_foods": [],
  "needs_confirmation_foods": [],
  "no_candidate_foods": [],
  "messages": [
    "일부 음식은 영양성분 계산 전 사용자 확인이 필요합니다.",
    "현재 총 영양성분은 자동 확정된 음식만 합산한 값입니다."
  ]
}
```

### POST `/api/v1/diets/analyses/{analysis_id}/foods/{food_item_id}/confirm`

Confirms a food item using one of the returned nutrition candidates.

Request:

```json
{
  "candidate_id": "food-001-candidate-01",
  "food_code": "P101-103000100-3619",
  "serving_amount": 1.0,
  "serving_unit": "serving"
}
```

Response `200 OK`:

```json
{
  "analysis_id": "diet-analysis-20260610-0001",
  "food_item_id": "food-001",
  "status": "partial",
  "food": {
    "food_item_id": "food-001",
    "vision_food_name": "파프리카",
    "display_name": "파프리카 크래커",
    "food_code": "P101-103000100-3619",
    "food_item_status": "confirmed_by_user",
    "nutrition": {
      "energy_kcal": 296.0,
      "carbohydrate_g": 36.0,
      "protein_g": 2.0,
      "fat_g": 16.0,
      "sodium_mg": 560.0
    }
  },
  "summary": {
    "nutrition_calculation_status": "partial",
    "total_energy_kcal": 1127.0
  }
}
```

### POST `/api/v1/diets/analyses/{analysis_id}/foods/{food_item_id}/manual`

Confirms a food item through manual food or nutrition input.

Request:

```json
{
  "food_name": "생선찌개",
  "serving_size": "1 bowl",
  "nutrition": {
    "energy_kcal": 180.0,
    "carbohydrate_g": 8.0,
    "protein_g": 22.0,
    "fat_g": 6.0,
    "sodium_mg": 720.0
  }
}
```

Response `200 OK`:

```json
{
  "analysis_id": "diet-analysis-20260610-0001",
  "food_item_id": "food-018",
  "status": "partial",
  "food": {
    "food_item_id": "food-018",
    "vision_food_name": "생선찌개",
    "display_name": "생선찌개",
    "food_item_status": "confirmed_by_user",
    "source": "manual",
    "nutrition": {
      "energy_kcal": 180.0,
      "carbohydrate_g": 8.0,
      "protein_g": 22.0,
      "fat_g": 6.0,
      "sodium_mg": 720.0
    }
  },
  "summary": {
    "nutrition_calculation_status": "partial"
  }
}
```

## 3. Response Status

- `processing`: analysis is still running.
- `completed`: all detected foods are confirmed and summary nutrition is complete.
- `needs_user_confirmation`: at least one food needs candidate selection or manual input.
- `partial`: some foods are confirmed, but one or more foods remain unresolved.
- `failed`: image analysis failed.

## 4. Food Item Status

- `matched`: normalized vision food matched one nutrition candidate and can be auto-confirmed.
- `weak_match`: a candidate exists but is likely a product, partial match, or broad mismatch.
- `multiple_candidates`: candidates exist but automatic selection is unsafe.
- `no_candidates`: nutrition provider returned no usable candidate.
- `no_query`: no lookup query was available.
- `api_unavailable`: nutrition provider failed or timed out.
- `manual_input_required`: user must manually search or enter nutrition.
- `confirmed_by_user`: user selected or manually entered a food.

## 5. JSON Response Examples

### Auto-Confirmed Item

```json
{
  "food_item_id": "food-002",
  "vision_food_name": "비빔국수",
  "display_name": "비빔국수",
  "source": "mfds",
  "food_code": "D303-157000000-0001",
  "serving_size": "100g",
  "nutrition": {
    "energy_kcal": 113.0,
    "carbohydrate_g": 20.0,
    "protein_g": 3.4,
    "fat_g": 2.1,
    "sodium_mg": 337.0
  },
  "editable": true,
  "user_action": "can_edit"
}
```

### Needs-Confirmation Item

```json
{
  "food_item_id": "food-001",
  "vision_food_name": "파프리카",
  "raw_food_name": "파프리카",
  "nutrition_status": "weak_match",
  "match_status": "weak_match",
  "candidates": [
    {
      "candidate_id": "food-001-candidate-01",
      "source": "mfds",
      "food_name": "파프리카 크래커",
      "food_code": "P101-103000100-3619",
      "match_status": "weak_match",
      "rank_score": 90.0,
      "rank_reason": "name_startswith_query; query_contained_in_name; token_overlap:파프리카; processed_food_penalty:-15",
      "serving_size": null,
      "nutrition_preview": {
        "energy_kcal": null,
        "carbohydrate_g": null,
        "protein_g": null,
        "fat_g": null,
        "sodium_mg": null
      }
    }
  ],
  "message": "정확한 영양성분 계산을 위해 가장 가까운 음식을 선택해주세요.",
  "editable": true,
  "user_action": "select_candidate"
}
```

### No-Candidate Item

```json
{
  "food_item_id": "food-018",
  "vision_food_name": "생선찌개",
  "raw_food_name": "생선찌개",
  "nutrition_status": "no_candidates",
  "message": "영양성분 후보를 찾지 못했습니다. 음식을 직접 입력하거나 다시 검색해주세요.",
  "editable": true,
  "user_action": "manual_search_required"
}
```

## 6. Summary Calculation Policy

- Include only `auto_confirmed` and `confirmed_by_user` foods in total nutrition.
- Exclude `needs_confirmation_foods` before user selection.
- Exclude `no_candidates` and `no_query` before manual input.
- `nutrition_calculation_status` values:
  - `completed`: all detected foods are confirmed.
  - `partial`: at least one detected food remains unresolved.
  - `unavailable`: no confirmed nutrition is available.

## 7. Frontend UI Policy

- Auto-confirmed section:
  - Show food cards with nutrition.
  - Allow user edits for food name, serving amount, and candidate replacement.
- Needs-confirmation section:
  - Show top 3 to 5 candidates.
  - Do not add nutrition to summary until the user confirms.
- No-candidate/manual section:
  - Show manual search or manual nutrition input.
- Summary:
  - Show a `partial` badge when unresolved food exists.
  - Recalculate after each confirm/manual action.

## 8. Backend Implementation Notes

- Auto-confirm only `matched`.
- Never auto-confirm `weak_match`, `multiple_candidates`, or `fallback_used`.
- Never use evaluation labels such as `expected_foods` in production lookup.
- Do not log API keys.
- Do not expose raw provider responses directly to clients.
- Return only top 3 to 5 nutrition candidates.
- Handle provider timeout as `api_unavailable`.
- Store enough data to recompute summary after user actions.

## 9. Implementation TODO

Functions likely needed when porting to `app/services/diets.py`:

- `analyze_diet_image(image) -> analysis_id`
- `extract_food_candidates(image) -> list[FoodCandidate]`
- `lookup_nutrition_candidates(food_names) -> list[NutritionCandidate]`
- `build_diet_analysis_response(analysis) -> DietAnalysisResponse`
- `confirm_food_candidate(analysis_id, food_item_id, candidate_id)`
- `apply_manual_food_input(analysis_id, food_item_id, payload)`
- `recalculate_nutrition_summary(analysis_id)`

DB fields likely needed:

- `analysis_id`
- `user_id`
- `image_path` or storage object key
- `analysis_status`
- `food_item_id`
- `vision_food_name`
- `raw_food_name`
- `food_item_status`
- `candidate_id`
- `food_code`
- `selected_food_name`
- nutrition values
- serving amount and unit
- user confirmation timestamp

Deferred work:

- Serving amount and portion correction
- Disease-specific diet score integration
- RDA or menu provider integration
- Provider cache and retry policy
- Admin monitoring for provider failures
