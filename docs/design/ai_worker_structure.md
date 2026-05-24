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
| `ai_worker/jobs/` | 향후 worker/job entrypoint 후보 영역. 현재 queue/stream 구현은 없음 |

현재 건강검진 OCR 공식 방향은 PaddleOCR/local OCR 1차이며, Clova OCR provider는 삭제하지 않고 PoC/deferred provider로 보존합니다. 공식 시연 경로와 demo readiness 검증에서는 Clova OCR 호출과 env 설정을 필수 조건으로 보지 않습니다.

현재 작업 범위는 파일 위치와 import 경로 정리입니다. Redis Stream, Celery, background worker, 신규 파이프라인 기능은 구현하지 않습니다.
