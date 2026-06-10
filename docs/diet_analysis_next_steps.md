# 식단 이미지 분석 이후 작업 정리

## 1. 현재 상태

식단 이미지 분석 기능은 현재 **실험 검증 완료 / production 연결 전 보류** 상태다.

배포 교육 및 배포 연습이 우선이므로, 식단 분석 production 연결 작업은 잠시 중단한다. 배포 흐름을 먼저 안정화한 뒤 mock API endpoint부터 다시 이어간다.

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

### 1) Mock API 먼저

- `POST /api/v1/diets/analyze-image`
- `GET /api/v1/diets/analyses/{analysis_id}`
- 실제 GPT/MFDS 호출 없이 fixture를 반환한다.
- 프론트가 UI 흐름을 먼저 붙일 수 있게 한다.

### 2) 프론트 UI 연결

- 자동 확정 카드
- 후보 선택 카드
- 후보 없음 직접 입력
- partial summary 배지

### 3) Production provider 이식

- GPT Vision food extraction
- MFDS lookup
- reranking
- service response builder
- `matched`만 자동 확정

### 4) Confirm / Manual API

- 후보 선택 확정
- 직접 입력 확정
- summary 재계산

### 5) Portion 보정

- `serving_amount`
- `portion_multiplier`
- 사용자 수정 여부

### 6) 질환 점수 연결

- 우선 `energy_kcal`, `carbohydrate_g`, `protein_g`, `fat_g`, `sodium_mg`만 사용한다.
- 이후 `sugar`, `saturated fat`, `potassium` 등은 고도화 단계에서 확장한다.

## 5. 지금 보류하는 이유

- 배포 교육 및 배포 연습이 우선이다.
- 식단 이미지 분석 기능은 실험/설계 단계가 충분히 정리되었다.
- production 연결은 배포 흐름을 먼저 안정화한 뒤 재개한다.
- 다시 이어갈 때는 mock API endpoint부터 시작한다.
