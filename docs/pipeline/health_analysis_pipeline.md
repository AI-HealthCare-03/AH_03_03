# 건강위험 분석 파이프라인

## 목적

사용자가 입력한 건강정보를 기준으로 당뇨 및 고혈압 위험도를 분석하고, 주요 위험요인과 추천 챌린지를 생성한다. Backend는 요청/조회/저장을 담당하고, AI Worker는 모델 입력 생성, 추론, SHAP 또는 feature contribution 계산을 담당한다.

## 입력값

- `user_id`: 분석 요청 사용자
- `health_record_id`: 분석 기준 건강정보
- 건강정보 주요 항목:
  - 키, 몸무게, BMI
  - 수축기/이완기 혈압
  - 공복혈당, HbA1c
  - 당뇨/고혈압 질환 여부
  - 흡연, 음주, 운동 빈도, 수면 시간

## 처리 흐름

1. 사용자가 건강정보를 입력한다.
2. `health_records`에 최신 건강정보를 저장한다.
3. 분석 요청 시 `async_jobs`를 생성한다.
4. `health_record_id` 기준으로 모델 입력 피처를 생성한다.
5. 당뇨 위험도와 고혈압 위험도를 계산한다.
6. SHAP 또는 feature contribution 값을 생성한다.
7. `analysis_results`에 최종 위험도 결과를 저장한다.
8. `analysis_result_factors`에 주요 위험요인을 저장한다.
9. `analysis_snapshots`에 입력/출력/SHAP 원본을 저장한다.
10. `challenge_recommendations`에 추천 챌린지를 저장한다.
11. 필요 시 `notifications`에 웹 내부 알림을 생성한다.
12. Dashboard API는 별도 테이블 없이 관련 테이블을 실시간 조회/집계한다.

## 출력 테이블

- `async_jobs`: 분석 작업 생성 및 상태 관리
- `analysis_results`: 당뇨/고혈압 위험도 최종 결과
- `analysis_result_factors`: 주요 위험요인 및 SHAP 기여도
- `analysis_snapshots`: 분석 입력/출력/SHAP 원본
- `challenge_recommendations`: 분석 결과 기반 추천 챌린지
- `notifications`: 분석 완료 또는 추천 생성에 대한 웹 내부 알림

## AI Worker와 Backend 역할 분리

Backend 담당:

- 사용자 인증 및 권한 확인
- 건강정보 입력/조회/수정 API
- 분석 요청 API
- `async_jobs` 생성
- 분석 결과 조회 API
- Dashboard 조회/집계 API
- 내부 알림 조회 API

AI Worker 담당:

- `async_jobs` 기반 분석 작업 처리
- `health_record_id` 기준 모델 입력 피처 생성
- ML 모델 추론
- SHAP 또는 feature contribution 계산
- 분석 결과, 위험요인, 스냅샷, 추천 챌린지 저장
- 작업 성공/실패 상태 갱신

## SHAP Factor 저장 기준

- 저장 대상은 사용자에게 설명 가능한 상위 위험요인으로 제한한다.
- `factor_key`는 코드에서 사용하는 안정적인 영문 키를 사용한다.
- `factor_name`은 화면 표시용 한글명을 사용한다.
- `factor_value`는 수치, 범주, 문장 값을 모두 담을 수 있도록 varchar로 저장한다.
- `contribution_score`는 SHAP 또는 feature contribution 값을 저장한다.
- `direction`은 위험도를 높이면 `POSITIVE`, 낮추면 `NEGATIVE`, 판단 불가 시 `NEUTRAL`을 사용한다.
- 화면 노출 순서는 `display_order`로 제어한다.

## Challenge Recommendation 생성 기준

- 위험도 높은 분석 결과를 우선 고려한다.
- SHAP factor 중 기여도가 큰 항목을 추천 근거로 사용한다.
- 혈압 관련 위험요인은 혈압/운동/생활습관 챌린지와 연결한다.
- 혈당 관련 위험요인은 혈당/식습관/운동 챌린지와 연결한다.
- 이미 참여 중이거나 최근 완료한 챌린지는 중복 추천하지 않는 것을 원칙으로 한다.
- MVP 1차에서는 정교한 개인화 랭킹보다 규칙 기반 추천을 우선하고, 이후 ML/LLM 기반 랭킹을 고도화한다.

## Dashboard 집계 기준

Dashboard는 별도 저장 테이블을 만들지 않고 다음 테이블을 실시간 조회/집계한다.

- 최근 건강정보: `health_records`
- 최신 위험도 결과: `analysis_results`
- 주요 위험요인: `analysis_result_factors`
- 챌린지 진행 상태: `user_challenges`
- 추천 챌린지: `challenge_recommendations`
- 읽지 않은 알림 수: `notifications`

## Nutrition Scoring 데이터 기준

식단 이미지 분석 이후 영양 DB와 매칭할 수 있도록 식품별 질병군 점수 테이블을 별도 모듈로 분리한다.

- 원본 엑셀 위치: `experiment/ai/cv/food/nutrition/raw/food_nutrition_db.xlsx`
- 런타임 점수 CSV: `ai_runtime/cv/food/nutrition/data/food_disease_scores.csv`
- 점수 규칙: `ai_runtime/cv/food/nutrition/rules/disease_score_rules.json`
- 스코어러: `ai_runtime/cv/food/nutrition/scoring/disease_food_scorer.py`

원본 엑셀은 서비스 런타임에서 직접 읽지 않는다. 원본 엑셀을 수정하거나 교체한 경우 아래 명령으로 런타임 CSV를 재생성한다.

```bash
uv run python -m ai_runtime.cv.food.nutrition.scoring.disease_food_scorer
```

생성되는 CSV는 `DM`, `HTN`, `DL`, `OBE`, `ANEM` 5개 질병군에 대해 0~100점 범위의 참고용 식품 적합도 점수를 포함한다. 높은 점수는 해당 질병군 관리 맥락에서 상대적으로 활용하기 쉬운 식품이라는 의미이며, 의료 진단이나 영양 처방이 아니다.

현재 `/api/v1/diets/analyze` 공식 경로는 자체 식단 CV 모델을 아직 호출하지 않는다. 대신 기존 음식명 후보 생성 흐름에서 나온 음식명을 `DiseaseFoodScorer`의 런타임 CSV와 매칭해 `disease_scores`, `food_score_details`, `scoring_source=nutrition_rule_table`을 응답과 `DietRecord.nutrition_summary`, `DietPhotoResult.raw_output`에 포함한다. 공식 저장 payload의 source는 `rule_based_food_detection`으로 기록하며 `rule_stub`, `image_analysis_stub` 같은 개발용 표현은 노출하지 않는다.

식단 이미지 provider 결과는 `ai_runtime/cv/food/schemas.py`의 `FoodDetectionCandidateSet` 형태로 정규화하는 방향을 기준으로 한다. provider 우선순위는 자체 CV 모델 -> GPT Vision -> rule-based food detection이다. 자체 CV confidence가 `CV_CONFIDENCE_THRESHOLD` 이상이고 음식명 후보가 충분하면 바로 nutrition scorer로 이동한다. confidence가 낮거나 음식명 후보가 부족하면 GPT Vision fallback 후보가 되지만, 비용 발생 API이므로 현재 정책은 `user_confirmation_required`이다. 어떤 provider를 쓰더라도 최종적으로 `detected_foods: list[str]`를 `DiseaseFoodScorer` 입력으로 넘기는 구조를 유지한다.

정규화 필드 기준:

- `provider`: `cv_model` | `gpt_vision` | `rule_based_food_detection`
- `confidence`: `float | null`
- `detected_foods`: `list[str]`
- `needs_review`: `bool`
- `fallback_reason`: `string | null`

## 건강검진 OCR Provider 정책

건강검진 OCR 공식 실행 경로는 GPT Vision/PaddleOCR 흐름을 사용한다.

- 기본 방향: PaddleOCR/local OCR 1차
- fallback 후보: GPT Vision
- GPT Vision fallback 기본값: off

설정 기준:

- `GPT_VISION_FALLBACK_ENABLED=false`
- `EXAM_OCR_PROVIDER=fallback`은 OCR provider 미설정 상태를 뜻하며, 더미 측정값을 저장하지 않는다.
- 운영에서 GPT Vision OCR을 사용하려면 `OPENAI_API_KEY`, `EXAM_OCR_PROVIDER=gpt_vision`, `EXAM_GPT_VISION_ENABLED=true`, `EXAM_GPT_VISION_MODEL`, `GPT_VISION_FALLBACK_ENABLED=true`를 `.prod.env`에 설정한다.

현재 `/api/v1/exams/{exam_id}/ocr`는 OCR 결과 확인/confirm 후 `ExamMeasurement` 값을 `HealthRecord` X2 필드에 반영하는 서비스 흐름을 검증하는 데 초점을 둔다.
인식 후보가 없거나 provider가 꺼져 있으면 OCR job은 실패 상태가 되어야 하며, `ExamMeasurement` 더미 row를 만들지 않는다. 측정값 조회 SQL에서는 FK 컬럼 `exam_report_id`를 사용한다.

## MVP 범위 기준

이번 프로젝트의 MVP는 풀서비스 1차 범위를 기준으로 하며, 소셜 로그인과 웨어러블 연동만 제외한다.

MVP에 포함한다:

- DIET 기록/분석 흐름
- LLM 응답 생성 및 결과 설명 흐름
- FAMILY, QNA, MEDICATION, ADMIN 기능
- 운영 로그와 관리자 모니터링
- 알림 예약/발송 이력 기반 구조

MVP 범위 안에서 후속 단계로 둔다:

- 외부 SMS/Email/Push/Kakao 발송 worker
- 실시간 스트리밍 분석
- 복잡한 모델 registry
- 독립적인 queue abstraction 설계
