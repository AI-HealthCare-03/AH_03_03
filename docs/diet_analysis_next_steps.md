# 식단 이미지 분석 이후 작업 정리

## 1. 현재 상태

식단 이미지 분석 기능은 현재 **기존 production 경로 운영 / 후보확정형 실험 계약 미연결** 상태다.

현재 production 경로는 `POST /api/v1/diets/analyze` async job이다. 이 경로는 이미지 업로드를 저장하고
`diet.analyze_image` job을 통해 `DietRecord`와 `DietPhotoResult`를 생성한다.

`experiment/ai/cv/gpt_vision_food_eval`과 `experiment/ai/cv/food_nutrition_api_eval`에 정리된 후보확정형 API
(`auto_confirmed_foods`, `needs_confirmation_foods`, `no_candidate_foods`, confirm/manual API)는 아직 production runtime에
연결하지 않았다.

## 2. 완료된 작업

- GPT Vision 음식 후보 추출 실험
- AI-Hub 음식 이미지 기반 평가
- GPT Vision prompt 개선
  - `empty_result_count`: 22/30 -> 0/30
  - `raw_food_names` 확보 row: 약 8/30 -> 30/30
- MFDS 식품영양성분 API 연결
- MFDS 후보 reranking 및 안전 처리
- `matched` / `weak_match` / `multiple_candidates` / `no_query` / `no_candidates` 상태 분리
- `expected_foods` label leakage 차단
- MFDS nutrition lookup provider 실험
- service response JSON 샘플 생성
- frontend mock response fixture 생성
- 식단 이미지 분석 API 계약서 초안 작성

## 3. Production 연결 시 핵심 정책

- `expected_foods`는 production에서 절대 사용하지 않는다.
- API key, log, raw response를 노출하지 않는다.
- MFDS top1을 그대로 자동 확정하지 않는다.
- `matched`만 자동 확정한다.
- `weak_match`, `multiple_candidates`, `fallback_used`는 사용자 확인 대상으로 둔다.
- `no_candidates`, `no_query`는 직접 검색 또는 직접 입력 대상으로 둔다.
- 확정된 음식만 nutrition summary에 합산한다.

## 4. 이후 재개 순서

### 1) 프론트 optional UI 준비

- 기존 `POST /api/v1/diets/analyze` async job 흐름은 유지한다.
- 기존 응답에 후보확정형 필드가 없어도 화면이 깨지지 않게 한다.
- 향후 응답에 아래 optional field가 있을 때만 후보 확인 섹션을 표시한다.
  - `auto_confirmed_foods`
  - `needs_confirmation_foods`
  - `no_candidate_foods`
  - `nutrition_calculation_status`
  - `needs_user_confirmation`

### 2) Service response builder

- GPT Vision food extraction 결과와 MFDS lookup/reranking 결과를 production DTO로 변환한다.
- `matched`만 자동 확정한다.
- `weak_match`, `multiple_candidates`, `fallback_used`는 사용자 확인 대상으로 둔다.
- `no_candidates`, `no_query`는 직접 검색 또는 직접 입력 대상으로 둔다.
- 확정된 음식만 nutrition summary에 합산한다.

### 3) Confirm / Manual API

- 후보 선택 확정
- 직접 입력 확정
- summary 재계산
- serving/portion 보정

### 4) MFDS provider feature flag 연결

- MFDS lookup provider를 production runtime에 feature flag 뒤로 연결한다.
- API key, raw response, cache, latency, fallback 정책을 운영 기준으로 검증한다.
- 장애 시 기존 `rule_based_food_detection`과 `nutrition_rule_table` 흐름을 유지할 수 있게 한다.

### 5) Portion 보정 고도화

- `serving_amount`
- `portion_multiplier`
- 사용자 수정 여부

### 6) 질환 점수 연결 고도화

- 우선 `energy_kcal`, `carbohydrate_g`, `protein_g`, `fat_g`, `sodium_mg`만 사용한다.
- 이후 `sugar`, `saturated fat`, `potassium` 등은 고도화 단계에서 확장한다.

## 5. 지금 보류하는 이유

- 배포 교육 및 배포 연습이 우선이다.
- 식단 이미지 분석 기능은 실험/설계 단계가 충분히 정리되었다.
- 후보확정형 API는 실험 계약과 mock fixture가 있지만 production runtime에는 아직 연결하지 않았다.
- 다시 이어갈 때는 기존 `/diets/analyze` async job을 깨지 않는 optional UI와 service response builder부터 시작한다.
