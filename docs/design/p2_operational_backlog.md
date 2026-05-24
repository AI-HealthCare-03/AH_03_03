# P2 Operational Backlog

이 문서는 `feature/kdu` 기준으로 시연 전에는 구현하지 않는 작업을 명확히 분리하기 위한 운영 백로그다. 아래 항목은 "미구현"이라기보다, 시연 안정성과 범위 통제를 위해 의도적으로 보류한 작업이다.

시연 전 P0의 초점은 동기 API 기반 핵심 흐름이다.

- 로그인/회원가입
- 건강정보 입력
- 건강검진 OCR confirm
- BASIC/PRECISION 분석
- CatBoost DM/HTN/DL 추론
- OBESITY rule-based 분석
- 식단 nutrition score
- 관리자 콘솔/FAQ/문의
- Docker compose 기반 FastAPI 구동

운영 진입 단계에서는 아래 P1/P2 항목을 순차적으로 구현한다.

## 1. 전체 항목 요약

| 번호 | 항목 | 우선순위 | 현재 상태 | 시연 전 P0 제외 사유 | 운영 진입 전 필요한 이유 |
| --- | --- | --- | --- | --- | --- |
| 1 | Redis Stream / async_jobs / AI Worker consumer | P2 | 구조 후보만 존재 | 시연은 동기 API로 충분하며 queue 도입 시 장애 지점이 늘어남 | OCR/CV/LLM 장시간 작업의 retry, status tracking, DLQ 필요 |
| 2 | vector RAG / pgvector embedding search | P2 | RAG-ready interface 수준 | 신뢰 문서 수집/임베딩/검증 없이 붙이면 의료 답변 위험 증가 | 근거 기반 건강정보 답변, 출처 추적, 검색 품질 개선 필요 |
| 3 | wearable 연동 | P2 | 요구사항 보류 | 외부 계정/OAuth/device API 범위가 크고 시연 핵심 흐름이 아님 | 연속 건강 데이터 기반 추세 분석과 리마인더 자동화에 필요 |
| 4 | 외부 알림 worker | P2 | reminder/log UI와 스키마 중심 | SMS/Push/Kakao/Email 발송은 비용/인증/실패처리 이슈가 큼 | 실제 리마인더 발송, 재시도, 발송 이력 관리 필요 |
| 5 | Langfuse 운영 추적 | P1 | 실험/연동 구조 존재 | 실제 LLM 호출 기본값이 꺼져 있어 시연 핵심 장애 요인이 아님 | LLM 비용, prompt version, safety/fallback 관측 필요 |
| 6 | Sentry/Datadog 등 모니터링 | P1 | DB 로그 중심 | 외부 SaaS 연동보다 현재 Docker/API 안정화가 우선 | 운영 장애 알림, trace, error aggregation 필요 |
| 7 | rate limiting | P1 | 로그인 제한 등 일부 정책만 존재 | 시연 로컬 환경에서는 abuse 방어보다 UX 검증이 우선 | 로그인/LLM/OCR/분석 API 남용 방지 필요 |
| 8 | DB backup / restore | P1 | PostgreSQL volume 중심 | 시연 DB는 seed 재현이 가능하고 백업 자동화는 운영 영역 | 운영 데이터 보호, 장애 복구, 롤백 절차 필요 |
| 9 | migration squash | P2 | 누적 migration 유지 | 시연 전 DB 변경 안정성이 더 중요하며 squash는 회귀 위험 있음 | 신규 운영 DB 초기 적용 시간 단축, migration history 정리 |
| 10 | load test | P2 | 미실행 | 시연은 단일/소수 사용자 흐름 검증이 우선 | 동시 사용자, ML 로딩, DB 커넥션 병목 검증 필요 |
| 11 | model registry 고도화 | P2 | artifact path + metadata JSON 중심 | 현재 DM/HTN/DL CatBoost artifact로 시연 가능 | 모델 버전 관리, 배포 승인, rollback, metric 추적 필요 |
| 12 | ML model warm-up 운영화 | P1 | 수동 warmup script 후보 | startup에 강제 연결하면 부팅 지연/실패 위험이 있음 | 첫 요청 지연 감소, artifact 상태 사전 검증 필요 |

## 2. 항목별 상세

### 1. Redis Stream / async_jobs / AI Worker consumer

- 현재 상태:
  - `ai_worker/jobs/`, `ai_worker/pipelines/`는 향후 구조 후보로 존재한다.
  - 실제 Redis Stream producer/consumer, `async_jobs` 테이블, worker orchestration은 구현하지 않는다.
- 왜 시연 전 P0가 아닌지:
  - 현재 핵심 시연 흐름은 FastAPI 동기 API로 검증 가능하다.
  - queue/worker를 넣으면 상태 전이, retry, timeout, idempotency까지 함께 설계해야 해 시연 리스크가 커진다.
- 운영 진입 전 필요한 이유:
  - OCR, CV, GPT Vision, LLM, 대용량 ML 추론은 요청 시간이 길어질 수 있다.
  - 작업 상태 조회, 실패 재시도, dead-letter queue, worker heartbeat가 필요하다.
- 예상 변경 파일/모듈:
  - `app/models/async_jobs.py`
  - `app/apis/v1/job_routers.py`
  - `ai_worker/jobs/`
  - `ai_worker/pipelines/`
  - Redis Stream producer/consumer 모듈
- 우선순위: P2

### 2. vector RAG / pgvector embedding search

- 현재 상태:
  - `ai_worker/llm`에 RAG-ready interface와 안전한 fallback 구조가 있다.
  - 실제 문서 수집, chunking, embedding, pgvector 검색은 구현하지 않는다.
- 왜 시연 전 P0가 아닌지:
  - 검증되지 않은 RAG는 의료 정보 답변 품질과 안전성 위험을 키운다.
  - 현재 시연은 rule-based 설명과 safety notice로 충분히 방어 가능하다.
- 운영 진입 전 필요한 이유:
  - 건강정보 답변에는 공신력 있는 출처와 근거 추적이 중요하다.
  - 향후 질병/영양/복약 설명을 근거 문서 기반으로 제공하려면 embedding 검색이 필요하다.
- 예상 변경 파일/모듈:
  - `ai_worker/llm/rag/`
  - `ai_worker/llm/rag_generator.py`
  - `ai_worker/llm/rag_sources.py`
  - pgvector migration/model
  - 문서 ingest script
- 우선순위: P2

### 3. wearable 연동

- 현재 상태:
  - 풀서비스 논의에서는 보류 항목으로 분리한다.
  - 현재 MVP 시연 범위에는 포함하지 않는다.
- 왜 시연 전 P0가 아닌지:
  - 외부 플랫폼 OAuth, 사용자 동의, device API, 데이터 주기 정책이 필요하다.
  - 핵심 분석 시연은 사용자가 입력한 건강정보와 OCR 검진정보로 가능하다.
- 운영 진입 전 필요한 이유:
  - 수면, 걸음 수, 심박, 운동량 같은 연속 데이터를 자동 수집할 수 있다.
  - 추적 대시보드와 챌린지 수행률 자동화를 고도화할 수 있다.
- 예상 변경 파일/모듈:
  - `app/apis/v1/wearable_routers.py`
  - `app/services/wearables.py`
  - `app/models/wearables.py`
  - external provider client
- 우선순위: P2

### 4. 외부 알림 worker

- 현재 상태:
  - 알림 inbox, reminder schedule, notification log 중심의 구조와 UI를 우선한다.
  - 실제 Push/SMS/Kakao/Email 발송 worker는 구현하지 않는다.
- 왜 시연 전 P0가 아닌지:
  - 외부 발송은 API key, 비용, 발송 실패, 재시도, 수신 동의 정책이 필요하다.
  - 시연에서는 예약/이력 관리와 in-app 알림 흐름 확인이 더 중요하다.
- 운영 진입 전 필요한 이유:
  - 복약, 챌린지, 건강기록 리마인더가 실제 사용자에게 전달되어야 한다.
  - channel별 실패 원인과 retry 정책이 필요하다.
- 예상 변경 파일/모듈:
  - `app/services/notifications.py`
  - `app/models/notifications.py`
  - `ai_worker/jobs/notification_worker.py`
  - SMS/Email/Push/Kakao provider
- 우선순위: P2

### 5. Langfuse 운영 추적

- 현재 상태:
  - Langfuse 실험/연동 구조는 존재한다.
  - 실제 LLM 호출은 기본값으로 켜지지 않는 흐름을 유지한다.
- 왜 시연 전 P0가 아닌지:
  - 시연 핵심은 rule-based/fallback 설명과 ML 분석 결과 확인이다.
  - 실제 LLM 호출 비용과 외부 API 상태에 시연 성공을 의존하지 않는다.
- 운영 진입 전 필요한 이유:
  - prompt version, model, token 사용량, fallback reason, safety result를 추적해야 한다.
  - 의료 표현 safety failure를 관측하고 개선할 수 있다.
- 예상 변경 파일/모듈:
  - `ai_worker/llm/llm_client.py`
  - `ai_worker/llm/llm_generator.py`
  - `ai_worker/llm/explanation_service.py`
  - Langfuse env/config
- 우선순위: P1

### 6. Sentry/Datadog 등 모니터링

- 현재 상태:
  - DB 기반 system error log, sensitive access log 중심이다.
  - 외부 APM/Sentry/Datadog 연동은 없다.
- 왜 시연 전 P0가 아닌지:
  - 로컬/시연 환경에서는 FastAPI logs, Docker logs, DB logs로 원인 파악이 가능하다.
  - 외부 SaaS 연동은 env 관리와 비용/권한 설정이 필요하다.
- 운영 진입 전 필요한 이유:
  - 운영 장애를 실시간으로 감지하고 알림을 받을 수 있어야 한다.
  - API latency, exception, slow query, 외부 provider 실패율 관측이 필요하다.
- 예상 변경 파일/모듈:
  - `app/core/monitoring.py`
  - `app/main.py`
  - `app/middlewares/`
  - Sentry/Datadog env config
- 우선순위: P1

### 7. rate limiting

- 현재 상태:
  - 인증 일부 흐름에는 제한 정책이 있으나, 전체 API rate limiting은 운영 수준으로 구현하지 않는다.
- 왜 시연 전 P0가 아닌지:
  - 시연은 제한된 사용자와 로컬 환경에서 진행된다.
  - rate limit을 성급히 넣으면 정상 클릭 흐름이 막힐 수 있다.
- 운영 진입 전 필요한 이유:
  - 로그인 brute force, OCR/GPT Vision/LLM API 남용, 분석 API 과호출을 막아야 한다.
  - 비용 발생 provider 보호가 필요하다.
- 예상 변경 파일/모듈:
  - `app/middlewares/rate_limit.py`
  - `app/core/config.py`
  - Redis-backed limiter
  - endpoint별 policy config
- 우선순위: P1

### 8. DB backup / restore

- 현재 상태:
  - Docker compose volume 기반 영속화가 중심이다.
  - 자동 `pg_dump`, restore runbook, retention 정책은 없다.
- 왜 시연 전 P0가 아닌지:
  - 시연 데이터는 seed로 재현 가능하다.
  - 백업 자동화는 운영 데이터 보호 정책과 함께 설계해야 한다.
- 운영 진입 전 필요한 이유:
  - 사용자 건강정보, 분석 결과, 가족/문의/알림 데이터 손실 방지가 필요하다.
  - 장애 발생 시 복구 절차와 RPO/RTO 기준이 필요하다.
- 예상 변경 파일/모듈:
  - `scripts/backup_postgres.sh`
  - `scripts/restore_postgres.sh`
  - `docs/ops/backup_restore.md`
  - deployment cron/job
- 우선순위: P1

### 9. migration squash

- 현재 상태:
  - 개발 과정의 migration history가 누적되어 있다.
  - 시연 전에는 기존 migration 흐름을 유지한다.
- 왜 시연 전 P0가 아닌지:
  - squash는 DB schema 재생성과 regression 위험이 있다.
  - 현재는 마이그레이션 안정성과 데이터 보존이 더 중요하다.
- 운영 진입 전 필요한 이유:
  - 신규 환경 초기화 시간을 줄이고 migration history를 이해하기 쉽게 만든다.
  - legacy drop/rename migration을 정리할 수 있다.
- 예상 변경 파일/모듈:
  - `migrations/`
  - `app/core/db/migrations/`
  - `docs/erd/mvp_erd.dbml`
  - DB bootstrap script
- 우선순위: P2

### 10. load test

- 현재 상태:
  - 단위/스모크 테스트 중심이다.
  - k6, Locust 등 부하 테스트는 없다.
- 왜 시연 전 P0가 아닌지:
  - 시연은 기능 흐름과 안정성 확인이 우선이다.
  - 부하 테스트는 infra sizing과 운영 목표가 정해진 뒤 의미가 커진다.
- 운영 진입 전 필요한 이유:
  - CatBoost 모델 로딩, OCR/CV/LLM 호출, DB connection pool 병목을 확인해야 한다.
  - 동시 사용자 수, p95 latency, timeout 기준이 필요하다.
- 예상 변경 파일/모듈:
  - `tests/load/`
  - `docs/ops/load_test_plan.md`
  - k6/Locust scenario
- 우선순위: P2

### 11. model registry 고도화

- 현재 상태:
  - CatBoost artifact는 `ai_worker/ml/artifacts/{dm,htn,dl}/catboost` 경로와 JSON metadata로 관리한다.
  - 별도 registry server나 model promotion workflow는 없다.
- 왜 시연 전 P0가 아닌지:
  - 현재 DM/HTN/DL 모델 artifact와 `feature_columns.json`, `threshold.json`, `metrics.json`으로 추론 가능하다.
  - registry 고도화는 학습/배포 운영 프로세스가 확정된 뒤 진행하는 편이 안전하다.
- 운영 진입 전 필요한 이유:
  - 모델 버전 승인, rollback, metric 비교, artifact integrity check가 필요하다.
  - disease별 model lifecycle을 추적해야 한다.
- 예상 변경 파일/모듈:
  - `ai_worker/ml/common/artifacts.py`
  - `ai_worker/ml/inference/catboost_predictor.py`
  - `docs/design/ml_model_registry_design.md`
  - model metadata schema
- 우선순위: P2

### 12. ML model warm-up 운영화

- 현재 상태:
  - 모델 warmup은 수동 스모크 스크립트나 별도 함수로 처리하는 방향이 안전하다.
  - FastAPI startup에서 항상 15개 fold 모델을 강제 로드하지 않는다.
- 왜 시연 전 P0가 아닌지:
  - startup 강제 warm-up은 컨테이너 부팅 시간을 늘리고, artifact 문제 시 서버 자체가 뜨지 않을 수 있다.
  - 시연 전 수동 warmup으로 첫 요청 지연을 줄이는 것이 더 안전하다.
- 운영 진입 전 필요한 이유:
  - 첫 PRECISION 분석 요청의 cold start 지연을 줄여야 한다.
  - artifact load 실패를 사용자 요청 전에 탐지할 수 있어야 한다.
- 예상 변경 파일/모듈:
  - `ai_worker/ml/inference/disease_risk_service.py`
  - `scripts/warmup_ml_models.py`
  - FastAPI lifespan optional hook
  - healthcheck extension
- 우선순위: P1

## 3. 시연 전 제외 원칙

아래 기준에 해당하는 작업은 시연 전 P0에서 제외한다.

- 외부 유료 API 호출을 기본값으로 켜야 하는 작업
- queue/worker/status/retry 설계가 함께 필요한 비동기 작업
- DB schema를 크게 바꾸거나 migration 위험이 큰 작업
- 의료 답변 품질 검증 없이 사용자에게 새 답변 경로를 노출하는 작업
- 운영 관측/백업/부하처럼 실제 배포 환경 기준이 먼저 필요한 작업

## 4. 다음 단계 제안

시연 이후에는 아래 순서로 진행한다.

1. P1 운영 안정화
   - ML model warm-up 운영화
   - rate limiting
   - DB backup/restore
   - Sentry/Datadog 또는 최소 alerting
   - Langfuse 운영 추적
2. P2 비동기/검색 고도화
   - Redis Stream / async_jobs / AI Worker consumer
   - external notification worker
   - vector RAG / pgvector embedding search
   - model registry 고도화
   - load test
   - migration squash
3. 장기 확장
   - wearable 연동
   - provider별 비용/쿼터 관리
   - 운영 데이터 기반 모델 재학습 파이프라인

## 5. 완료 기준

이 문서의 목적은 보류 항목을 명확히 설명하는 것이다.

- 시연 범위와 운영 진입 범위가 구분되어 있다.
- 각 항목에 현재 상태와 제외 사유가 적혀 있다.
- 운영 진입 전 필요한 이유와 예상 변경 파일이 적혀 있다.
- Redis Stream, vector RAG, wearable, 외부 알림 worker 등이 "빠진 기능"이 아니라 "의도적으로 P2 보류한 기능"으로 설명 가능하다.
