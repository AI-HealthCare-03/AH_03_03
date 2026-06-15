# AI Worker Structure

`ai_runtime/`는 서비스에서 사용하는 AI 기능별 실행 영역입니다. develop 브랜치의 담당자별 `ai_worker/` 코드와 의미 충돌을 줄이기 위해 서비스 런타임 패키지명을 `ai_runtime`으로 분리했습니다.

| 경로 | 역할 |
| --- | --- |
| `ai_runtime/common/` | 여러 AI 영역에서 공유할 유틸, 공통 schema, 파일/입력 검증 후보 영역 |
| `ai_runtime/ml/` | CatBoost/XGBoost 학습/추론, X2 룰 기반 fallback, 모델 artifact |
| `ai_runtime/ocr/` | 건강검진표 OCR, OCR extractor/parser, Clova OCR PoC/deferred provider |
| `ai_runtime/cv/` | 음식 이미지 분석 라우터, 이미지 분석 평가 스크립트, CV 도메인 schema |
| `ai_runtime/llm/` | 일반 LLM 호출, GPT Vision 호출 계층, 프롬프트, RAG 준비, 상담/해설 생성 |
| `ai_runtime/pipelines/` | OCR → ML → LLM처럼 여러 AI 모듈을 묶는 orchestration 후보 영역 |
| `ai_runtime/jobs/` | Redis Stream consumer, job handler registry, retry/DLQ, pending recovery, scheduler |

현재 건강검진 OCR 공식 방향은 `EXAM_OCR_PROVIDER=auto`입니다. PDF는 측정값 페이지를 우선 선택하고 PaddleOCR을 먼저 시도하며, 실패하거나 후보가 없으면 PyMuPDF로 페이지 이미지를 만든 뒤 GPT Vision fallback을 사용할 수 있습니다. 이미지는 GPT Vision 우선, PaddleOCR fallback 정책입니다. Clova OCR provider는 삭제하지 않고 PoC/deferred provider로 보존합니다.

## 현재 비동기 처리 범위

- FastAPI 라우터와 Tortoise/asyncpg 기반 DB I/O는 async 기반입니다.
- Redis는 compose infrastructure, `/api/v1/system/health` 연결 확인, Redis Stream job queue 용도입니다.
- `ai_runtime/main.py`는 Redis Stream consumer와 scheduler loop를 실행합니다.
- `ai_runtime/jobs/redis_stream.py`는 AI stream, service stream, DLQ stream, MAXLEN, retry/backoff, pending recovery를 담당합니다.
- `ai_runtime/jobs/consumer.py`는 `XREADGROUP` 처리와 실패 시 retry/DLQ 이동, `XAUTOCLAIM` 기반 pending 회수를 담당합니다.
- `ai_runtime/jobs/handlers.py`는 job type별 handler registry를 관리합니다.

현재 주요 handler는 아래와 같습니다.

- `DEMO_ECHO`
- `analysis.run`
- `exam_ocr.run`
- `diet.analyze_image`
- `email.verification.send`
- `password_reset.email.send`
- `family.invite.email.send`
- `fcm.push.send`
- `family.notification.create`

프론트 분석 화면은 `/analysis/run-async`를 사용합니다. 기존 `/analysis/run` 동기 실행 API는 410 Gone으로 막아 긴 분석이 요청 중 직접 실행되지 않게 합니다. 건강검진 OCR과 식단 분석은 202 Accepted로 job을 만들고 `/api/v1/jobs/{job_id}` polling으로 상태를 확인합니다. 복약 정보는 MVP에서 OCR 없이 직접 입력합니다.

운영 고도화 과제로는 worker heartbeat/metrics 노출, 관리자용 queue 모니터링, DLQ 재처리 도구, worker 수평 확장 정책이 남아 있습니다.
