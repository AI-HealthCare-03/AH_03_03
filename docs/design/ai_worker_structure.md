# AI Worker Structure

`ai_worker/`는 서비스에서 사용하는 AI 기능별 실행 영역입니다. 새 최상위 `AI_worker` 폴더를 만들지 않고, 아래 도메인 단위로 책임을 나눕니다.

| 경로 | 역할 |
| --- | --- |
| `ai_worker/common/` | 여러 AI 영역에서 공유할 유틸, 공통 schema, 파일/입력 검증 후보 영역 |
| `ai_worker/ml/` | CatBoost/XGBoost 학습/추론, X2 룰 기반 fallback, 모델 artifact |
| `ai_worker/ocr/` | 건강검진표 OCR, OCR extractor/parser, Clova OCR PoC/deferred provider |
| `ai_worker/cv/` | 음식 이미지 분석 라우터, 이미지 분석 평가 스크립트, CV 도메인 schema |
| `ai_worker/llm/` | 일반 LLM 호출, GPT Vision 호출 계층, 프롬프트, RAG 준비, 상담/해설 생성 |
| `ai_worker/pipelines/` | OCR → ML → LLM처럼 여러 AI 모듈을 묶는 orchestration 후보 영역 |
| `ai_worker/jobs/` | Redis Stream 기반 최소 job skeleton. 현재 `DEMO_ECHO` job만 처리 |

현재 건강검진 OCR 공식 방향은 PaddleOCR/local OCR 1차이며, Clova OCR provider는 삭제하지 않고 PoC/deferred provider로 보존합니다. 공식 시연 경로와 demo readiness 검증에서는 Clova OCR 호출과 env 설정을 필수 조건으로 보지 않습니다.

## 현재 비동기 처리 범위

- FastAPI 라우터와 Tortoise/asyncpg 기반 DB I/O는 async 기반입니다.
- Redis는 compose infrastructure, `/api/v1/system/health` 연결 확인, `DEMO_ECHO` Redis Stream skeleton 용도입니다.
- OCR/CV/ML/LLM workflow는 현재 기존 동기 API 흐름 안에서 처리합니다.
- `ai_worker/main.py`는 Redis Stream consumer를 실행하지만 현재 `DEMO_ECHO` job만 처리합니다.
- `AnalysisResult.async_job_id`는 향후 실제 분석 job과 `async_jobs` 테이블을 연결하기 위한 reserved field이며, 현재 `/analysis/run`과 연결되어 있지 않습니다.

현재 async job skeleton은 시연 안정성 확인용이며, 기존 `/analysis/run`, `/diets/analyze`, OCR confirm 경로는 동기 API 흐름을 유지합니다.

운영 전에는 `async_jobs` 모델, Redis Stream producer/consumer, consumer group, retry/dead-letter queue, worker heartbeat, idempotency key를 한 묶음으로 설계해야 합니다.
