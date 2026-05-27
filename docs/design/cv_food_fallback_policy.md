# 식단 CV / GPT Vision Fallback 정책

이 문서는 식단 분석에서 음식명 후보를 만드는 provider 선택 정책을 정리한다. 실제 유료 GPT Vision API 호출을 기본값으로 켜지 않으며, 현재 공식 경로의 기본 동작은 `rule_based_food_detection`과 nutrition scorer다.

## 1. 현재 상태

- 자체 식단 CV 모델은 아직 공식 서비스 경로에 연결되어 있지 않다.
- `/api/v1/diets/analyze`는 현재 rule-based 음식 후보를 생성한다.
- 생성된 음식명 후보는 `DiseaseFoodScorer`로 전달되어 `DM / HTN / DL / OBE / ANEM` 질병군별 식단 점수를 계산한다.
- `ai_runtime/cv/providers/gpt_vision.py`는 존재하지만 공식 식단 fallback으로 자동 호출하지 않는다.
- 실제 유료 GPT Vision 호출은 env flag와 사용자 확인 정책이 정리된 뒤 켜야 한다.

## 2. Provider 공통 결과 schema

식단 provider 결과는 아래 개념으로 정규화한다.

| 필드 | 의미 |
| --- | --- |
| `provider` | `cv_model`, `gpt_vision`, `rule_based_food_detection` 중 하나 |
| `detected_foods` | nutrition scorer에 전달할 음식명 후보 목록 |
| `confidence` | provider가 산출한 전체 신뢰도 또는 평균 신뢰도 |
| `needs_review` | 사용자 확인 또는 다른 provider 보강이 필요한지 |
| `fallback_reason` | fallback 후보가 된 이유 |
| `raw_output` | provider 원본 또는 정책 판단 근거 |

구현 위치:

- `ai_runtime/cv/food/schemas.py`
- `ai_runtime/cv/food/fallback_policy.py`

## 3. 선택 정책

1. 자체 CV 모델 결과가 있고 `confidence >= threshold`이면 CV 결과를 사용한다.
2. 자체 CV 모델 결과가 없으면 기본값으로 rule-based 음식 후보를 사용한다.
3. 자체 CV 모델 결과가 있지만 `confidence < threshold`이면 `needs_review=True`로 표시한다.
4. `GPT_VISION_FALLBACK_ENABLED=true`일 때만 GPT Vision fallback 후보를 만들 수 있다.
5. GPT Vision fallback 후보가 만들어져도 현재 정책에서는 실제 API를 자동 호출하지 않는다.
6. 유료 호출이 필요한 경우 사용자 확인 또는 명시적 운영 설정을 거쳐야 한다.
7. 최종적으로 `DiseaseFoodScorer` 입력은 음식명 후보 list로 통일한다.

## 4. 환경변수

| 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `GPT_VISION_FALLBACK_ENABLED` | `false` | GPT Vision fallback 후보 생성을 허용할지 |
| `FOOD_CV_CONFIDENCE_THRESHOLD` | `0.75` | 자체 CV 결과를 그대로 사용할 최소 confidence |

기본값에서는 GPT Vision API가 호출되지 않는다.

## 5. 현재 `/diets/analyze` 동작

현재 공식 식단 분석 경로는 아래 흐름이다.

```text
/api/v1/diets/analyze
→ rule_based_food_detection
→ FoodDetectionCandidateSet
→ DiseaseFoodScorer
→ disease_scores / food_score_details / scoring_source 저장 및 응답
```

저장 위치:

- `DietRecord.nutrition_summary`
- `DietPhotoResult.confidence_payload`
- `DietPhotoResult.raw_output`

## 6. 후속 작업

- 자체 CV 모델 inference 결과를 `FoodDetectionCandidateSet`으로 변환한다.
- 낮은 confidence 결과에 대해 사용자 확인 UI 또는 GPT Vision fallback 호출 버튼을 설계한다.
- GPT Vision 결과를 동일 schema로 정규화한다.
- 비용, timeout, retry, provider 로그 정책을 운영 설정으로 분리한다.
