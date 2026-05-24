# AI Worker 모델 및 AI 기능 범위

이 문서는 현재 `ai_worker` 하위 AI 기능을 시연/운영 설명 기준으로 구분한다. 핵심은 "로컬 모델 artifact로 직접 추론하는 기능"과 "rule-based 로직, 외부 provider API, parser skeleton, P2 보류 기능"을 분리해서 설명하는 것이다.

## 1. 요약

| 구분 | 현재 범위 | 시연 설명 기준 |
| --- | --- | --- |
| 로컬 모델 artifact | DM/HTN/DL CatBoost 3종 | FastAPI 컨테이너에 포함되어 정밀분석에서 사용 가능 |
| Rule-based 로직 | OBESITY, ANEM 참고 분류, X2 classifier, 식단 nutrition scorer | 모델 파일 없이 규칙/점수표 기반으로 동작 |
| 외부 provider/API 코드 | GPT Vision, Clova OCR, OpenAI LLM | GPT Vision/OpenAI는 정책과 env에 따라 활성화 후보, Clova OCR은 현재 공식 실행 경로에서 제외된 deferred provider |
| Skeleton/parser | 약봉투 OCR parser, 건강검진 OCR 처리 구조, 처방전 OCR 준비 구조 | MVP에서는 구조와 parser 중심, 실제 provider 연결은 일부 미완성 |
| P2 보류 | 자체 식단 CV 모델, vector RAG, Redis Stream/async_jobs/consumer, LLM/RAG 운영 고도화 | 시연 전 미구현이 아니라 운영 확장 단계로 의도적 보류 |

LLM/RAG의 공식 runtime 범위는 [LLM/RAG Runtime Scope](llm_runtime_scope.md)를 기준으로 설명한다. 현재 공식 API에서 직접 호출되는 LLM runtime은 분석/식단 설명 생성 중심이며, 메인 챗봇 LLM 라우터와 추천 문구 모듈은 준비됐지만 아직 공식 runtime에 직접 연결되지 않은 영역으로 구분한다.

## 2. 로컬 모델 Artifact 있음

| 기능 | 질환/코드 | 경로 | 현재 상태 | 비고 |
| --- | --- | --- | --- | --- |
| CatBoost 만성질환 예측 | DM / DIABETES | `ai_worker/ml/artifacts/dm/catboost` | 모델 fold 5개와 메타 JSON 존재 | 정밀분석에서 CatBoost 사용 대상 |
| CatBoost 만성질환 예측 | HTN / HYPERTENSION | `ai_worker/ml/artifacts/htn/catboost` | 모델 fold 5개와 메타 JSON 존재 | 정밀분석에서 CatBoost 사용 대상 |
| CatBoost 만성질환 예측 | DL / DYSLIPIDEMIA | `ai_worker/ml/artifacts/dl/catboost` | 모델 fold 5개와 메타 JSON 존재 | 정밀분석에서 CatBoost 사용 대상 |

각 artifact 디렉터리는 다음 파일 구조를 기준으로 한다.

- `model_fold1.cbm` ~ `model_fold5.cbm`
- `feature_columns.json`
- `threshold.json`
- `metrics.json`
- `model_params.json`
- `experiment_config.json`

서비스 입력값은 DB 필드명을 모델 feature명에 직접 맞추지 않고, `ai_worker/ml/inference/feature_mapper.py`에서 CatBoost 학습 feature schema로 변환한다.

## 3. 모델 파일은 없지만 Rule-based 로직 있음

| 기능 | 경로 | 공식 결과 저장 여부 | 현재 상태 |
| --- | --- | --- | --- |
| OBESITY 분석 | `app/services/analysis.py`, `ai_worker/ml/X2/health_stage_classifier.py` | `AnalysisResult` 대상 | 현재 별도 CatBoost artifact가 없어 `rule_based`로 처리 |
| ANEM 참고 분류 | `ai_worker/ml/X2/health_stage_classifier.py`, 식단 점수 체계 | 공식 `AnalysisResult` 대상 아님 | X2/식단 참고 분류로만 사용 |
| X2 health stage classifier | `ai_worker/ml/X2/health_stage_classifier.py` | 일부 분석/참고 로직에서 사용 | 건강 단계/질환군 참고 분류용 rule-based 로직 |
| 식단 nutrition scorer | `ai_worker/cv/food/nutrition/` | 식단 분석 응답/저장 payload에 포함 | DM/HTN/DL/OBE/ANEM 질병군별 음식 점수 계산 |

ANEM을 공식 분석 결과에 포함하려면 `AnalysisType` enum, DB schema, API DTO, UI, 테스트를 함께 확장해야 한다.

## 4. 외부 Provider/API 코드

| 기능 | 경로 | 현재 상태 | 주의사항 |
| --- | --- | --- | --- |
| GPT Vision 식단 이미지 분석 후보 | `ai_worker/cv/providers/gpt_vision.py` | provider 코드 존재 | 기본값은 off이며 `GPT_VISION_FALLBACK_ENABLED=true`와 사용자 확인 정책이 있을 때만 fallback 후보 |
| Clova OCR provider | `ai_worker/ocr/providers/clova_ocr/` | PoC/deferred provider로 보존 | 공식 건강검진 OCR 시연 경로에서는 호출하지 않으며 `ENABLE_CLOVA_OCR=false`가 기본 |
| OpenAI LLM | `ai_worker/llm/` | 챗봇, 설명 생성, fallback/rewrite 구조 존재 | 기본은 rule-based/fallback 설명이며 실제 LLM 호출은 설정에 따라 제한적으로 사용 |

외부 provider는 API key, 비용, 개인정보 처리 정책이 연결되므로 기본 시연 경로에서 무조건 호출하지 않는다.

건강검진 OCR 공식 방향은 PaddleOCR/local OCR 1차 처리와 GPT Vision fallback 후보 구조다. Clova OCR은 삭제하지 않고 legacy/PoC provider로 보존하지만, 시연 준비 검증이나 공식 API 기본 경로의 필수 조건으로 보지 않는다.

## 5. Skeleton / Parser

| 기능 | 경로 | 현재 상태 | 시연 설명 기준 |
| --- | --- | --- | --- |
| 약봉투 OCR parser | `ai_worker/ocr/medication/` | raw text parser와 schema 중심 구조 | 실제 OCR provider 호출 없이 parser skeleton으로 준비 |
| 건강검진 OCR 처리 구조 | `ai_worker/ocr/checkup/`, `app/services/exams.py` | OCR 결과 confirm 후 `HealthRecord` X2 필드 반영 흐름 존재 | provider 연결/문서 인식 품질은 별도 검증 필요 |
| 처방전 OCR | `ai_worker/ocr/` 및 medication domain 확장 후보 | 실제 provider 연결 미완성 | P1/P2에서 약봉투/처방전 스키마를 분리 고도화 |

## 6. P2 보류

| 항목 | 현재 상태 | 보류 사유 |
| --- | --- | --- |
| 자체 식단 CV 모델 | 미구현 | 현재는 음식명 후보 기반 nutrition scorer가 MVP 범위 |
| Vector RAG / pgvector embedding search | 준비 구조 중심 | 신뢰 문서 수집, chunking, embedding, retrieval 검증이 필요 |
| Redis Stream / async_jobs / AI Worker consumer | 미구현 | 현재 MVP는 동기 처리, 운영 확장 시 비동기 worker로 전환 |
| 실제 LLM/RAG 운영 연결 고도화 | 일부 interface/fallback 존재 | 비용, 안전성, grounding, 관측 정책 정리 후 운영화 |

## 7. 발표용 한 줄 정리

현재 `ai_worker`에서 로컬 모델 artifact로 직접 추론하는 것은 DM/HTN/DL CatBoost 3종이다. OBESITY와 ANEM 관련 흐름은 rule-based 또는 참고 분류이며, 식단 CV/GPT Vision/OCR/LLM은 provider 코드와 skeleton을 갖춘 상태에서 시연 기본 경로는 비용 없는 rule-based/fallback 중심으로 동작한다.

## 8. AI Worker 기능 감사 스크립트

현재 로컬 환경에서 AI Worker 기능 연결 상태를 확인하려면 아래 스크립트를 실행한다.

```bash
uv run python scripts/audit_ai_worker_capabilities.py
```

이 스크립트는 다음 항목을 점검한다.

- DM/HTN/DL CatBoost artifact 경로와 `.cbm` fold 파일 개수
- CatBoost predictor warmup 가능 여부
- X2 health stage classifier import 가능 여부
- 식단 nutrition scorer import 가능 여부와 runtime CSV record 개수
- GPT Vision, Clova OCR, OpenAI LLM provider 코드 import 가능 여부
- `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` 설정 여부
- Clova OCR이 deferred provider로 표시되는지 여부
- LLM prompt 관련 코드 위치와 line number
- LLM/RAG 공식 runtime, keyword RAG PoC, provider-only, prepared-not-wired, P2 backlog 분류
- 약봉투 OCR parser sample parse 가능 여부
- 건강검진 OCR checkup extractor import 가능 여부

주의사항:

- 외부 API를 실제 호출하지 않는다.
- 환경변수 값과 API key 원문은 출력하지 않는다.
- LLM prompt 원문 전문은 출력하지 않고 파일 경로와 line number만 출력한다.
- 감사 결과는 "현재 구현/Provider 코드/Backlog 상태"를 함께 보여준다. `NOT_IMPLEMENTED`, `P1_BACKLOG`, `P2_BACKLOG`는 전체 프로젝트 실패가 아니라 MVP 범위에서 의도적으로 보류한 항목일 수 있다.
- `READY_*` 항목이 많더라도 자체 식단 CV 모델, vector RAG, Redis Stream 기반 worker 같은 운영 확장 항목이 완료되었다는 뜻은 아니다.
- `READY_RUNTIME`은 현재 공식 API에서 실제 호출되는 경로를 뜻한다. `PREPARED_NOT_WIRED`는 코드가 준비되어도 공식 runtime에 아직 연결되지 않았다는 뜻이다.
- CatBoost 모델 로드 시간이 부담되면 아래처럼 warmup을 생략할 수 있다.

```bash
uv run python scripts/audit_ai_worker_capabilities.py --skip-warmup
```
