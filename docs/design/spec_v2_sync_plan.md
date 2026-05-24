# Spec v2 Sync Plan

## 1. 현재 진실 공급원

외부 `메인프로젝트 문서/`의 ERD/API/요구사항 ver1은 2026-05-19 기준 산출물로 보고, 현재 코드와 시연 기준의 진실 공급원은 내부 문서를 우선한다.

현재 기준 문서:

- `docs/design/`
  - `full_service_scope.md`
  - `ai_worker_service_integration_plan.md`
  - `ai_worker_structure.md`
  - `admin_access_design.md`
  - `admin_monitoring_design.md`
  - `family_service_design.md`
  - `challenge_recommendation_design.md`
  - `schema_normalization_plan.md`
- `docs/erd/mvp_erd.dbml`
- `docs/data/challenges/team_challenge_master.csv`

정리 기준:

- 외부 ver1 문서는 초기 기획 기준이다.
- 내부 `docs/design`과 `mvp_erd.dbml`은 2026-05-22~24 기준 풀서비스 1차 범위와 실제 구현 상태를 더 잘 반영한다.
- 제출/발표 전 외부 문서를 ver2로 갱신해 코드, ERD, API 명세, 발표 범위를 같은 기준으로 맞춘다.

## 2. ver2 업데이트 대상

| 외부 문서 | 업데이트 목적 | 내부 기준 |
| --- | --- | --- |
| 요구사항 정의서 | 풀서비스 1차 범위와 보류 범위를 다시 명시 | `docs/design/full_service_scope.md`, `docs/design/requirements_refactor_notes.md` |
| API 명세서 | 실제 FastAPI path, request/response, deprecated endpoint 정책 반영 | OpenAPI, `app/apis/v1/*`, `frontend/src/api/*` |
| ERD | 현재 DB 모델, role, analysis mode, family/admin/notification 구조 반영 | `docs/erd/mvp_erd.dbml` |
| Architecture drawio | FastAPI, frontend, postgres, redis, ai_worker, Docker, Langfuse 분리 구조 반영 | `docs/design/ai_worker_structure.md`, Docker compose 구조 |
| 발표용 기능 범위 표 | 구현 완료/부분 구현/보류 항목을 명확히 구분 | `docs/design/ai_worker_service_integration_plan.md`, 실제 코드 상태 |

## 3. ver2 반영 항목

### 인증/보안

- Firebase/social login은 현재 범위에서 제외한다.
- 인증은 JWT 기반으로 정리한다.
- 비밀번호 해싱은 Argon2id 기준으로 반영한다.
- refresh token cookie의 HttpOnly, SameSite, Secure, Path 정책을 명시한다.
- role은 5단계로 정리한다.
  - `USER`
  - `MONITOR`
  - `OPERATOR`
  - `ADMIN`
  - `SUPER_ADMIN`

### 사용자/운영 기능

- 가족 기능을 ver2 범위에 반영한다.
  - family group
  - family member
  - invite
  - share setting
- 관리자 콘솔을 반영한다.
  - summary
  - monitoring
  - system error log
  - sensitive access log
  - FAQ 관리
  - 1:1 문의 답변
- 알림/리마인더를 반영한다.
  - inbox notification
  - reminder schedule
  - notification log
  - 외부 Push/SMS/Kakao/Email 발송 worker는 후속 범위로 구분한다.

### 건강 분석

- `analysis_mode`를 반영한다.
  - `BASIC`: X1 기본 건강정보 기반 간편 분석
  - `PRECISION`: X1 + X2 검진/혈액검사 수치 기반 정밀 분석
- 건강검진 OCR confirm 흐름을 반영한다.
  - 건강검진표 OCR 실행
  - `ExamMeasurement` 저장
  - `/exams/{exam_id}/confirm`
  - `HealthRecord` X2 필드 반영
  - readiness `precision_ready=true`
  - `/analysis/run` `mode=PRECISION`
- CatBoost 모델 적용 범위를 반영한다.
  - `DM / DIABETES`: CatBoost
  - `HTN / HYPERTENSION`: CatBoost
  - `DL / DYSLIPIDEMIA`: CatBoost
  - `OBE / OBESITY`: 현재 rule-based
- 분석 결과 저장 항목을 반영한다.
  - `analysis_mode`
  - `model_name`
  - `model_version`
  - `risk_score`
  - `risk_level`
  - factors/snapshot

### 식단/CV/영양 점수

- 자체 식단 CV 모델은 아직 공식 분석 엔진으로 완성되지 않았음을 명시한다.
- 현재 `/diets/analyze`는 음식명 후보 기반 nutrition scorer를 연결한 상태로 정리한다.
- 질병군별 식단 점수 대상:
  - `DM`
  - `HTN`
  - `DL`
  - `OBE`
  - `ANEM`
- 런타임 기준:
  - `ai_worker/cv/food/nutrition/data/food_disease_scores.csv`
  - `ai_worker/cv/food/nutrition/rules/disease_score_rules.json`
- 원본 엑셀은 런타임에서 직접 읽지 않는다.
  - `etc/ai_worker/cv/food/nutrition/raw/food_nutrition_db.xlsx`
- GPT Vision은 fallback provider 후보로 반영하되, 실제 자동 호출은 보류 또는 사용자 확인 후 호출 정책으로 구분한다.

### 챌린지/데이터

- 팀 챌린지 master를 공식 기준으로 반영한다.
  - `docs/data/challenges/team_challenge_master.csv`
- 챌린지 유형을 반영한다.
  - `SPECIAL`
  - `COMMON`
  - `GENERAL`
- 질환군별 추천 대상, 주의 문구, 금기 문구를 반영한다.

### 보류/P2 인프라

아래 항목은 ver2에서 운영 진입용 P2로 명확히 보류 처리한다.

- Redis Stream
- `async_jobs`
- 별도 AI Worker consumer
- retry/dead-letter queue
- idempotency key
- worker status monitoring
- RAG vector DB 고도화

## 4. 제출/발표 시 주의 문구

- 외부 ver1 문서만 기준으로 평가하면 현재 코드와 불일치할 수 있다.
- 최종 제출/발표 기준 문서는 ver2로 갱신해야 한다.
- ver2의 기준은 내부 `docs/design`, `docs/erd/mvp_erd.dbml`, 실제 OpenAPI를 우선한다.
- 구현 상태는 과장하지 않는다.
  - 완료
  - 부분 구현
  - 구조만 존재
  - 보류
  - 후속 작업
- 의료 관련 표현은 진단/확진/처방이 아니라 위험도/참고용/관리 필요 신호로 설명한다.

## 5. 권장 업데이트 순서

1. 외부 요구사항 정의서 ver2 작성
2. 외부 ERD를 `docs/erd/mvp_erd.dbml` 기준으로 재생성
3. OpenAPI 기준으로 API 명세서 ver2 작성
4. Architecture drawio 갱신
5. 발표용 기능 범위 표 갱신
6. 발표 자료에 ver1/ver2 차이와 보류 범위를 짧게 명시

## 6. 완료 기준

- 외부 요구사항/API/ERD/Architecture/발표 표가 내부 기준과 같은 범위를 사용한다.
- Firebase/social login, wearable 연동은 제외 또는 후속 범위로 명시한다.
- Redis Stream/async_jobs/worker orchestration은 P2 보류로 명시한다.
- CatBoost DM/HTN/DL, OCR confirm -> X2, BASIC/PRECISION, 가족/관리자/알림/식단 점수 구조가 ver2에 반영된다.
