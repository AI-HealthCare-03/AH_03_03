# LLM/RAG Runtime Scope

이 문서는 `feature/kdu` 기준 LLM/RAG 코드의 공식 runtime 경로와 준비/PoC/legacy 후보를 구분한다. 코드 이동 리팩터링 없이 현재 호출 관계를 설명하는 문서다.

## 1. 현재 결론

현재 공식 API에서 직접 쓰는 LLM 계층은 챗봇 전체가 아니라 **분석/식단 결과 설명 생성** 중심이다. 기본 동작은 외부 OpenAI 호출이 없는 rule-based explanation이며, keyword RAG PoC context와 Langfuse trace metadata가 보조로 연결되어 있다.

| 분류 | 상태 | 설명 |
| --- | --- | --- |
| 공식 runtime | 사용 중 | `app/services/analysis.py`, `app/services/diets.py`가 `ai_worker.llm.explanation_service`를 호출 |
| Keyword RAG PoC | 사용 중 | `docs/rag_sources` markdown source를 keyword/disease_type으로 조회하고 reference source를 설명에 첨부 |
| Langfuse trace | 사용 가능 | RAG retrieval metadata를 no-op 안전 구조로 기록. env 미설정 시 외부 호출 없음 |
| 챗봇 LLM 라우터 | 준비됨, 미연결 | `response_router.py`, `health_chatbot.py`, `rule_engine.py`, `llm_generator.py`는 공식 챗봇 API에 아직 직접 연결되지 않음 |
| 추천/챌린지 문구 | 준비됨, 일부 미연결 | `recommendation_message.py`, `risk_mapper.py`는 구조가 있으나 공식 분석 경로는 DB challenge recommendation 중심 |
| Vector RAG/LangChain | P2 보류 | embedding, pgvector retrieval, LangChain/LangGraph는 구현하지 않음 |

## 2. 공식 Runtime 경로

### 만성질환 분석 결과 설명

호출 경로:

```text
app/services/analysis.py
  -> ai_worker.llm.explanation_service.retrieve_health_context()
  -> ai_worker.llm.rag.retrieve_keyword_rag_contexts()
  -> ai_worker.llm.rag.tracing.trace_keyword_rag_retrieval()
  -> ai_worker.llm.explanation_service.generate_explanation_with_context()
```

역할:

- `AnalysisResult`의 `analysis_type`, `risk_score`, `risk_level`, `model_name`, `model_version`, factor를 입력으로 받는다.
- 기본 설명은 `rule_based_explanation`이다.
- keyword RAG context가 있으면 `reference_summary`, `reference_sources`를 붙인다.
- `safety_notice`는 항상 유지한다.
- Langfuse 설정이 없거나 `LANGFUSE_ENABLED=false`이면 trace 기록은 no-op이다.

### 식단 점수 설명

호출 경로:

```text
app/services/diets.py
  -> ai_worker.cv.food.nutrition.scoring.DiseaseFoodScorer
  -> ai_worker.llm.explanation_service.generate_diet_score_explanation()
```

역할:

- `DM`, `HTN`, `DL`, `OBE`, `ANEM` 질병군별 식단 점수를 받아 가장 낮은 점수 중심으로 짧은 설명을 만든다.
- 기본 설명은 `rule_based_explanation`이다.
- 실제 OpenAI 호출은 하지 않는다.
- 식단 설명에도 의료 진단/처방이 아니라 참고용 건강관리 안내라는 안전 문구를 포함한다.

## 3. 공통 모듈

| 파일 | 역할 | 현재 분류 |
| --- | --- | --- |
| `ai_worker/llm/schemas.py` | 챗봇, 추천문구, 분석 설명, RAG context 공통 Pydantic schema | 공통 |
| `ai_worker/llm/safety.py` | 의료 안전 문구/위험 표현 검사 | 공통 |
| `ai_worker/llm/llm_client.py` | OpenAI 호출, Langfuse trace/event helper | 공통 provider adapter |
| `ai_worker/llm/prompt_templates.py` | 추천문구, 챗봇, RAG prompt template | 공통 prompt |

`schemas.py`는 여러 기능 schema가 한 파일에 모여 있다. 시연 전에는 유지하고, 운영 전 규모가 커지면 `schemas/chatbot.py`, `schemas/explanation.py`, `schemas/rag.py`로 나누는 것을 검토한다.

## 4. RAG 모듈

### 공식 PoC 경로

| 파일 | 역할 | 현재 분류 |
| --- | --- | --- |
| `ai_worker/llm/rag/source_loader.py` | `docs/rag_sources/index.json`과 markdown source 로드 | READY_RAG_POC |
| `ai_worker/llm/rag/keyword_retriever.py` | disease_type/keyword 기반 source 선택 | READY_RAG_POC |
| `ai_worker/llm/rag/rag_context_builder.py` | `RetrievedContext`, reference summary/source 변환 | READY_RAG_POC |
| `ai_worker/llm/rag/tracing.py` | Langfuse trace metadata 생성/기록 | READY_RAG_POC |

주의:

- vector DB, embedding, pgvector retrieval은 구현하지 않는다.
- markdown 원문은 짧은 요약/candidate source 중심이다.
- 모든 source status는 운영 승인 전까지 `candidate_unreviewed` 또는 별도 검토 상태로 관리한다.

### Legacy/PoC 후보

| 파일 | 역할 | 현재 분류 |
| --- | --- | --- |
| `ai_worker/llm/rag_generator.py` | 이미 검색된 context를 받아 main health RAG 답변 생성 | legacy/PoC 후보 |
| `ai_worker/llm/rag_sources.py` | 허용 RAG source/domain whitelist | legacy/PoC 후보 |

이 두 파일은 현재 공식 API runtime에서 직접 호출되지 않는다. 삭제하지 말고, 운영형 RAG 설계 시 유지/통합 여부를 결정한다.

## 5. 아직 공식 Runtime이 아닌 준비 모듈

| 파일 | 현재 역할 | 현재 분류 |
| --- | --- | --- |
| `ai_worker/llm/response_router.py` | rule engine, LLM fallback/rewrite 라우팅 | PREPARED_NOT_WIRED |
| `ai_worker/llm/health_chatbot.py` | 메인/결과 기반 챗봇 rule-based 응답 | PREPARED_NOT_WIRED |
| `ai_worker/llm/rule_engine.py` | 챗봇 intent/rule 처리 | PREPARED_NOT_WIRED |
| `ai_worker/llm/llm_generator.py` | OpenAI 기반 생성, stub fallback, rewrite | PREPARED_NOT_WIRED |
| `ai_worker/llm/grounding.py` | 결과 기반 챗봇 grounding 검사 | PREPARED_NOT_WIRED |

현재 공식 챗봇 API인 `app/services/chatbot.py`는 아직 `ai_worker.llm.response_router`를 직접 호출하지 않는다. 지금은 앱 서비스 내부의 간단 rule-based 응답 중심이다.

향후 챗봇 정렬 방향:

```text
app/services/chatbot.py
  -> ai_worker.llm.response_router.route_main_health_chatbot_response()
  -> ai_worker.llm.response_router.route_result_chatbot_response()
```

이 전환 전에는 API 응답 DTO, 안전 문구, grounding 실패 fallback, 실제 LLM 호출 flag를 함께 검증해야 한다.

## 6. 추천/챌린지 모듈

| 파일 | 현재 역할 | 현재 분류 |
| --- | --- | --- |
| `ai_worker/llm/risk_mapper.py` | ML 예측 결과를 위험요인/추천 챌린지 후보로 변환 | PREPARED_NOT_WIRED |
| `ai_worker/llm/recommendation_message.py` | 위험요인/챌린지 기반 사용자 문구 생성 | PREPARED_NOT_WIRED |

현재 공식 분석 서비스는 `app/services/analysis.py`에서 DB의 active challenge를 조회해 recommendation을 만든다. 위 모듈은 구조는 준비되어 있지만 공식 분석 결과 생성 경로의 핵심 호출부는 아니다.

## 7. P2 보류

| 항목 | 상태 | 이유 |
| --- | --- | --- |
| Vector RAG / pgvector embedding search | P2_BACKLOG | 신뢰 문서 수집, chunking, embedding, retrieval 검증이 먼저 필요 |
| LangChain / LangGraph | P2_BACKLOG | 현재 PoC는 간단 keyword retrieval로 충분하며 운영 orchestration 도입 전 검토 필요 |
| Redis Stream / async_jobs | P2_BACKLOG | 현재 MVP는 동기 처리. 장시간 OCR/CV/ML/LLM 작업을 운영화할 때 도입 |
| 운영형 LLM/RAG 평가 파이프라인 | P2_BACKLOG | Langfuse trace는 준비됐지만, 평가 dataset/metric/review workflow는 별도 설계 필요 |

## 8. Audit 분류 기준

`scripts/audit_ai_worker_capabilities.py`는 LLM/RAG 항목을 아래 기준으로 표시한다.

| Category | 의미 |
| --- | --- |
| `READY_RUNTIME` | 공식 API에서 현재 호출되는 runtime 경로 |
| `READY_RAG_POC` | keyword RAG PoC로 동작하며 외부 API 없이 검증 가능한 경로 |
| `READY_PROVIDER_CODE_ONLY` | provider/client 코드는 있으나 env/정책에 따라 선택적으로 활성화 |
| `PREPARED_NOT_WIRED` | 구조는 준비됐지만 공식 API runtime에 직접 연결되지 않음 |
| `P2_BACKLOG` | MVP 시연 전 의도적 보류 항목 |

## 9. 권장 리팩터링 판단

시연 전에는 파일 이동 리팩터링을 하지 않는다. 현재 import는 `app/services/analysis.py`, `app/services/diets.py`, `tests/llm`, `scripts/audit_ai_worker_capabilities.py`에 걸려 있어 폴더 이동 시 깨질 가능성이 있다.

최소 순서:

1. 이 문서와 audit 출력으로 공식 runtime 범위를 고정한다.
2. 챗봇 공식 API를 `response_router`로 연결할지 별도 작업에서 결정한다.
3. 운영 전 `schemas.py` 분리 여부를 검토한다.
4. `rag_generator.py`, `rag_sources.py`는 legacy/PoC 후보로 문서화한 뒤 유지/통합 여부를 결정한다.
