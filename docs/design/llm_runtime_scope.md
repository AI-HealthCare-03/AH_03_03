# LLM/RAG Runtime Scope

이 문서는 `feature/kdu-llm` 기준 LLM/RAG 코드의 공식 runtime 경로와 준비/PoC/legacy 후보를 구분한다. 상세 LangGraph node/edge 구조는 `docs/design/llm_langgraph_runtime_plan.md`를 기준으로 본다.

## 1. 현재 결론

현재 공식 API에서 쓰는 LLM 계층은 **LangGraph 기반 챗봇 runtime**, **분석/식단 결과 설명 생성**, **keyword RAG reference source 연결**로 나뉜다. 기본 동작은 외부 OpenAI 호출이 없는 rule/fallback 경로이며, 실제 LLM rewrite/generation은 env flag와 provider key가 있을 때만 선택적으로 사용한다.

| 분류 | 상태 | 설명 |
| --- | --- | --- |
| 공식 runtime | 사용 중 | `app/services/chatbot.py`가 LangGraph runner를 호출하고, 분석/식단 설명은 `explanation_service`와 analysis explanation node를 사용 |
| Keyword RAG runtime | 사용 중 | `docs/rag_sources` markdown source를 keyword/disease_type으로 조회하고 `reference_sources`/`reference_summary`를 생성 |
| Langfuse trace | 사용 가능 | graph/node metadata, prompt version, retrieval id, source trust level을 no-op 안전 구조로 기록. env 미설정 시 외부 호출 없음 |
| 챗봇 LLM 라우터 | 사용 중 | `response_router.py`, `health_chatbot.py`, `rule_engine.py`, `llm_generator.py`가 LangGraph generation node 내부에서 호출됨 |
| 추천/챌린지 action | 부분 구현 | `build_recommended_actions` node가 안전 수준과 context 기반 문자열 action을 만들고, 구조화 metadata는 trace에 남김 |
| Vector RAG/pgvector | P2 보류 | embedding, pgvector retrieval, 운영형 source registry는 후속 확장 영역 |

## 2. 공식 Runtime 경로

### 메인 건강 챗봇

호출 경로:

```text
app/services/chatbot.py
  -> ai_runtime.llm.graph.run_chatbot_graph()
  -> normalize_input
  -> check_mental_health_safety
  -> classify_intent
  -> retrieve_rag_context
  -> generate_llm_answer
  -> check_grounding_or_fallback
  -> build_recommended_actions
  -> format_final_response
```

역할:

- 위기 키워드는 LLM generation/rewrite를 우회하고 safety response를 우선한다.
- 일반 건강 질문은 keyword RAG 또는 rule 기반 답변을 생성한다.
- RAG source가 없거나 약하면 grounding/fallback 문구를 더 보수적으로 만든다.
- API 응답 contract는 `answer`, `source`, `context_type`, `recommended_actions`, `safety_notice`를 유지한다.

### 만성질환 분석 결과 설명

호출 경로:

```text
app/services/analysis.py
  -> ai_runtime.llm.explanation_service.retrieve_health_context()
  -> ai_runtime.llm.rag.retrieve_keyword_rag_contexts()
  -> ai_runtime.llm.rag.tracing.trace_keyword_rag_retrieval()
  -> ai_runtime.llm.explanation_service.generate_explanation_with_context()
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
  -> ai_runtime.cv.food.nutrition.scoring.DiseaseFoodScorer
  -> ai_runtime.llm.explanation_service.generate_diet_score_explanation()
```

역할:

- `DM`, `HTN`, `DL`, `OBE`, `ANEM` 질병군별 식단 점수를 받아 가장 낮은 점수 중심으로 짧은 설명을 만든다.
- 기본 설명은 `rule_based_explanation`이다.
- 실제 OpenAI 호출은 하지 않는다.
- 식단 설명에도 의료 진단/처방이 아니라 참고용 건강관리 안내라는 안전 문구를 포함한다.

## 3. 공통 모듈

| 파일 | 역할 | 현재 분류 |
| --- | --- | --- |
| `ai_runtime/llm/schemas.py` | 챗봇, 추천문구, 분석 설명, RAG context 공통 Pydantic schema | 공통 |
| `ai_runtime/llm/safety.py` | 의료 안전 문구/위험 표현 검사 | 공통 |
| `ai_runtime/llm/llm_client.py` | OpenAI 호출, Langfuse trace/event helper | 공통 provider adapter |
| `ai_runtime/llm/prompt_templates.py` | 추천문구, 챗봇, RAG prompt template | 공통 prompt |

`schemas.py`는 여러 기능 schema가 한 파일에 모여 있다. 시연 전에는 유지하고, 운영 전 규모가 커지면 `schemas/chatbot.py`, `schemas/explanation.py`, `schemas/rag.py`로 나누는 것을 검토한다.

## 4. RAG 모듈

### 공식 PoC 경로

| 파일 | 역할 | 현재 분류 |
| --- | --- | --- |
| `ai_runtime/llm/rag/source_loader.py` | `docs/rag_sources/index.json`과 markdown source 로드 | READY_RAG_POC |
| `ai_runtime/llm/rag/keyword_retriever.py` | disease_type/keyword 기반 source 선택 | READY_RAG_POC |
| `ai_runtime/llm/rag/rag_context_builder.py` | `RetrievedContext`, reference summary/source 변환 | READY_RAG_POC |
| `ai_runtime/llm/rag/retriever.py` | retriever interface, keyword adapter, retrieval result contract | READY_RAG_POC |
| `ai_runtime/llm/rag/source_trust.py` | source type을 trust level로 분류 | READY_RAG_POC |
| `ai_runtime/llm/rag/tracing.py` | Langfuse trace metadata 생성/기록 | READY_RAG_POC |

주의:

- vector DB, embedding, pgvector retrieval은 아직 붙이지 않는다.
- markdown 원문은 짧은 요약/candidate source 중심이다.
- 모든 source status는 운영 승인 전까지 `candidate_unreviewed` 또는 별도 검토 상태로 관리한다.
- RAG 문서 본문 전체는 Langfuse trace metadata에 남기지 않는다.
- `source_trust_level`은 grounding check에서 응답 단정성을 조절하는 참고값으로 사용한다.

### 보조 모듈

| 파일 | 역할 | 현재 분류 |
| --- | --- | --- |
| `ai_runtime/llm/rag_generator.py` | 이미 검색된 context를 받아 main health RAG 답변 생성 | 사용 중 |
| `ai_runtime/llm/rag_sources.py` | 허용 RAG source/domain whitelist | 사용 중 |

두 파일은 LangGraph generation/RAG path에서 사용할 수 있으므로 삭제하지 않는다. 운영형 RAG 설계 시 prompt registry, whitelist, source trust policy와 함께 정리한다.

## 5. 아직 공식 Runtime이 아닌 준비 모듈

| 파일 | 현재 역할 | 현재 분류 |
| --- | --- | --- |
| `ai_runtime/llm/response_router.py` | rule engine, LLM fallback/rewrite 라우팅 | WIRED_BY_GRAPH |
| `ai_runtime/llm/health_chatbot.py` | 메인/결과 기반 챗봇 rule-based 응답 | WIRED_BY_GRAPH |
| `ai_runtime/llm/rule_engine.py` | 챗봇 intent/rule 처리 | WIRED_BY_GRAPH |
| `ai_runtime/llm/llm_generator.py` | OpenAI 기반 생성, stub fallback, rewrite | WIRED_BY_GRAPH |
| `ai_runtime/llm/grounding.py` | 결과 기반 챗봇 grounding 검사 | PREPARED_NOT_WIRED |

현재 공식 챗봇 API인 `app/services/chatbot.py`는 LangGraph runner를 호출하고, graph generation node 내부에서 `response_router`/`rag_generator`를 사용한다. 기존 API 응답 DTO와 safety notice contract는 유지한다.

## 6. 추천/챌린지 모듈

| 파일 | 현재 역할 | 현재 분류 |
| --- | --- | --- |
| `ai_runtime/llm/risk_mapper.py` | ML 예측 결과를 위험요인/추천 챌린지 후보로 변환 | PREPARED_NOT_WIRED |
| `ai_runtime/llm/recommendation_message.py` | 위험요인/챌린지 기반 사용자 문구 생성 | PREPARED_NOT_WIRED |

현재 공식 분석 서비스는 `app/services/analysis.py`에서 DB의 active challenge를 조회해 recommendation을 만든다. 위 모듈은 구조는 준비되어 있지만 공식 분석 결과 생성 경로의 핵심 호출부는 아니다.

## 7. P2 보류

| 항목 | 상태 | 이유 |
| --- | --- | --- |
| Vector RAG / pgvector embedding search | P2_BACKLOG | 신뢰 문서 수집, chunking, embedding, retrieval 검증이 먼저 필요 |
| LangChain 고도화 | P2_BACKLOG | LangGraph orchestration은 1차 연결됨. LangChain은 node 내부 부품 중심으로 제한 유지 |
| 운영형 LLM 비동기 처리 확대 | P2_BACKLOG | 분석/OCR/식단 async job은 존재하며, LLM 장시간 작업 분리는 후속 검토 |
| 운영형 LLM/RAG 평가 파이프라인 | P2_BACKLOG | Langfuse trace는 준비됐지만, 평가 dataset/metric/review workflow는 별도 설계 필요 |

## 8. Audit 분류 기준

`scripts/audit_ai_runtime_capabilities.py`는 LLM/RAG 항목을 아래 기준으로 표시한다.

| Category | 의미 |
| --- | --- |
| `READY_RUNTIME` | 공식 API에서 현재 호출되는 runtime 경로 |
| `READY_RAG_POC` | keyword RAG PoC로 동작하며 외부 API 없이 검증 가능한 경로 |
| `READY_PROVIDER_CODE_ONLY` | provider/client 코드는 있으나 env/정책에 따라 선택적으로 활성화 |
| `PREPARED_NOT_WIRED` | 구조는 준비됐지만 공식 API runtime에 직접 연결되지 않음 |
| `P2_BACKLOG` | MVP 시연 전 의도적 보류 항목 |

## 9. 권장 리팩터링 판단

시연 전에는 파일 이동 리팩터링을 하지 않는다. 현재 import는 `app/services/analysis.py`, `app/services/diets.py`, `tests/llm`, `scripts/audit_ai_runtime_capabilities.py`에 걸려 있어 폴더 이동 시 깨질 가능성이 있다.

최소 순서:

1. 이 문서와 audit 출력으로 공식 runtime 범위를 고정한다.
2. 챗봇 공식 API를 `response_router`로 연결할지 별도 작업에서 결정한다.
3. 운영 전 `schemas.py` 분리 여부를 검토한다.
4. `rag_generator.py`, `rag_sources.py`는 legacy/PoC 후보로 문서화한 뒤 유지/통합 여부를 결정한다.
