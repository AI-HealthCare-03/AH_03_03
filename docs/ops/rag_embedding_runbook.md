# RAG Embedding Runbook

이 문서는 RAG chunk embedding을 생성하고 저장할 때의 안전 절차를 정리한다. 실제 secret 값은 문서에 적지 않는다.

## 원칙

- RAG embedding은 식단 추천 API, LangGraph, 챗봇과 별도 단계로 운영한다.
- `mock` embedding은 개발/테스트 검증용이다.
- 운영 환경에서는 `mock` embedding apply를 사용하지 않는다.
- OpenAI embedding 저장 전에는 반드시 dry-run으로 대상 chunk 수와 batch 수를 확인한다.
- embedding vector 전문, chunk content 전문, API key는 로그나 trace에 남기지 않는다.

## 환경변수

예시 이름만 참고한다. 실제 값은 로컬 `.env` 또는 운영 secret store에만 둔다.

Docker compose로 실행할 때는 `fastapi`와 `ai-worker` service의 `environment`에 아래 값들이 전달되어야 한다. compose에 빠져 있으면 컨테이너 안의 `app.core.config` module-level 상수는 안전 기본값만 보게 된다.

기본 운영값은 keyword RAG를 유지하고 embedding/vector는 비활성화한다.

```env
RAG_ENABLED=true
RAG_RETRIEVAL_STRATEGY=keyword_only
MAIN_CHATBOT_RAG_STRATEGY=keyword_only
ANALYSIS_EXPLANATION_LLM_REWRITE_ENABLED=false
DIET_RECOMMENDATION_RAG_STRATEGY=keyword_only
RAG_EMBEDDING_ENABLED=false
RAG_EMBEDDING_PROVIDER=disabled
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_DIMENSION=1536
RAG_EMBEDDING_BATCH_SIZE=64
```

로컬에서 hybrid/vector RAG를 수동 검증할 때만 아래처럼 전환한다.

```env
MAIN_CHATBOT_RAG_STRATEGY=keyword_first_vector_fallback
RAG_RETRIEVAL_STRATEGY=keyword_first_vector_fallback
DIET_RECOMMENDATION_RAG_STRATEGY=keyword_first_vector_fallback
RAG_EMBEDDING_ENABLED=true
RAG_EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=<openai-api-key>
```

production 기본값은 `RAG_RETRIEVAL_STRATEGY=keyword_only`, `MAIN_CHATBOT_RAG_STRATEGY=keyword_only`, `DIET_RECOMMENDATION_RAG_STRATEGY=keyword_only`, `ANALYSIS_EXPLANATION_LLM_REWRITE_ENABLED=false`, `RAG_EMBEDDING_ENABLED=false`, `RAG_EMBEDDING_PROVIDER=disabled`로 유지한다. 실제 `.env`와 secret 값은 커밋하지 않는다.

Langfuse trace를 확인하려면 Langfuse 설정도 필요하다.

```env
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=<langfuse-url>
LANGFUSE_PUBLIC_KEY=<langfuse-public-key>
LANGFUSE_SECRET_KEY=<langfuse-secret-key>
```

## Dry-run

먼저 DB write 없이 대상 chunk와 예상 작업량을 확인한다.

FastAPI Docker image에는 `scripts/rag` 폴더가 포함되어 있어야 한다. dev compose 컨테이너에서 운영 스크립트를 실행할 때는 Makefile target을 우선 사용한다.

```bash
make rag-preview
make rag-ingest-dry-run
make rag-embed-dry-run
```

```bash
uv run python scripts/rag/embed_rag_chunks.py --provider openai --json
```

일부 문서만 확인하려면:

```bash
uv run python scripts/rag/embed_rag_chunks.py --provider openai --only-source-key hypertension --json
uv run python scripts/rag/embed_rag_chunks.py --provider openai --only-document-key rag:hypertension:hypertension.md --json
```

확인할 항목:

- `total_candidate_chunks`
- `planned_embedding_writes`
- `batches`
- `estimated_char_count`
- `warnings`
- `db_write_performed=false`

## Apply

dry-run 결과를 확인한 뒤에만 apply를 실행한다.

RAG chunk를 DB에 저장하려면:

```bash
make rag-ingest-apply
```

OpenAI embedding 대상과 비용을 소량으로 먼저 확인하려면:

```bash
make rag-embed-apply-openai-dry-run LIMIT=1
```

확인 후 실제 vector 저장을 소량 적용하려면:

```bash
make rag-embed-apply-openai LIMIT=1
```

```bash
uv run python scripts/rag/embed_rag_chunks.py --provider openai --apply --json
```

## Vector Search 수동 검증

embedding 저장 후 pgvector 검색이 동작하는지 read-only로 확인한다.

```bash
make rag-vector-query QUERY="당뇨 식단 주의사항" TOP_K=3
```

주의:

- OpenAI query embedding 비용이 소량 발생할 수 있다.
- 이 명령은 DB write를 하지 않는다.

## 현재 서비스 연결 표현

현재 구현 상태는 아래처럼 설명한다.

- 메인 챗봇은 LangGraph 기반 orchestration을 사용한다.
- RAG는 keyword retrieval을 기본으로 하며, feature flag로 `keyword_first_vector_fallback` 또는 `hybrid_parallel`을 사용할 수 있다.
- 식단 추천의 핵심 판단은 rule-based로 유지하고 LLM은 사용자-facing 문구 보조 역할을 한다.
- 분석 결과 설명은 rule-based 설명을 기본으로 하며, feature flag가 켜진 경우에만 LLM rewrite가 문장 정리를 보조한다.

RAG retrieval strategy 이름은 아래 의미로만 사용한다.

- `keyword_only`: keyword retriever만 사용한다.
- `keyword_first_vector_fallback`: keyword 결과가 부족할 때만 vector retriever를 fallback으로 사용한다.
- `hybrid_parallel`: keyword/vector를 모두 검색한 뒤 중복 제거와 단순 가중합 rerank로 최종 결과를 고른다.

아래 표현은 피한다.

- end-to-end LangChain RAG
- 완전한 hybrid RAG
- LLM이 식단/질환 추천을 판단
- LangGraph 기반 식단 추천 에이전트
- 전체 embedding apply 전에는 저장된 chunk가 적어 검색 결과가 제한적일 수 있다.
- 기본 출력은 chunk content 전문이 아니라 preview만 보여준다.

실패 시 즉시 중단하려면:

```bash
uv run python scripts/rag/embed_rag_chunks.py --provider openai --apply --fail-fast --json
```

모델 또는 dimension 변경 후 전체 재생성이 필요하면:

```bash
uv run python scripts/rag/embed_rag_chunks.py --provider openai --apply --force --json
```

## 재생성 기준

아래 조건이면 embedding은 stale로 보고 재생성 대상이 된다.

- `embedding`이 없음
- `embedding_content_hash != content_hash`
- `embedding_provider`가 현재 provider와 다름
- `embedding_model`이 현재 model과 다름
- `embedding_dimension`이 현재 dimension과 다름
- `--force` 사용

따라서 기존 `mock` embedding row는 OpenAI provider 기준으로 stale 처리되어 재생성된다.

## 운영 주의

- 운영 환경에서 `--provider mock --apply`는 금지한다.
- `make rag-ingest-apply`는 DB write를 수행한다.
- `make rag-embed-apply-openai`는 DB write와 OpenAI API 비용이 발생할 수 있다.
- OpenAI apply target은 기본 `LIMIT=1`로 동작한다. 대량 적용 전에는 반드시 소량 검증과 로그 확인을 먼저 한다.
- 비용과 rate limit을 고려해 batch size를 조절한다.
- OpenAI key가 없으면 OpenAI provider 생성은 실패해야 한다.
- pgvector index는 별도 단계다. chunk 수가 커지면 vector search 성능을 재검토한다.
- vector/hybrid retriever를 서비스 API에 연결하는 작업은 embedding 저장과 검색 품질 확인 이후에 진행한다.
- `.env`, API key, Langfuse secret은 커밋하지 않는다.

## Langfuse 확인

embedding trace에는 다음 정도만 남긴다.

- provider
- model
- dimension
- batch size
- chunk count
- failed count
- estimated char count
- apply 여부
- latency
- fallback reason

남기지 않는 것:

- chunk content 전문
- embedding vector 전문
- API key/secret

## 메인 챗봇 RAG Runtime Smoke

`/api/v1/chatbot/ask`에서 실제 RAG strategy와 LLM 답변 후처리를 확인할 때는 소량 요청만 실행한다.

권장 확인 순서:

```bash
docker compose --env-file .env -f infra/docker/docker-compose.dev.yml exec fastapi \
  uv run --no-sync python -c "from app.core import config; print('OPENAI_API_KEY_SET=', bool(config.OPENAI_API_KEY)); print('RAG_RETRIEVAL_STRATEGY=', config.RAG_RETRIEVAL_STRATEGY); print('RAG_EMBEDDING_ENABLED=', config.RAG_EMBEDDING_ENABLED); print('RAG_EMBEDDING_PROVIDER=', config.RAG_EMBEDDING_PROVIDER)"
```

그 다음 요청은 1~2건만 호출한다.

```bash
curl -fsS -X POST http://localhost:8080/api/v1/chatbot/ask \
  -H 'Content-Type: application/json' \
  -d '{"message":"당뇨 식단 관리는 어떻게 하면 좋나요?","context_type":"MAIN"}'
```

JWT 인증이 필요한 실제 runtime smoke는 토큰 값을 출력하지 않고 env로만 전달한다.

```bash
uv run python scripts/qa/smoke_chatbot_hybrid_rag_runtime.py \
  --base-url http://localhost:8080 \
  --token-env CHATBOT_SMOKE_TOKEN \
  --confirm-openai-call \
  --output-json reports/qa/chatbot_hybrid_rag_smoke_outputs.json \
  --output-md reports/qa/chatbot_hybrid_rag_smoke_outputs.md
```

로컬/dev 로그에서는 `main_chatbot_rag_retrieval` 레코드의 아래 값을 확인한다.

- `rag_strategy`
- `keyword_returned_count`
- `vector_returned_count`
- `merged_count`
- `final_count`
- `fallback_used`
- `fallback_reason`

금지:

- `docker compose config | grep OPENAI_API_KEY`처럼 secret 원문이 출력될 수 있는 명령을 실행하지 않는다.
- OpenAI key, Langfuse secret, embedding vector 전문을 로그나 문서에 남기지 않는다.

## Rollback/복구

embedding 저장은 RAG chunk 원문과 metadata를 바꾸지 않는다. 문제가 있으면 같은 script를 올바른 provider/model/dimension으로 다시 실행해 재생성한다.

운영에서 잘못 저장된 `mock` embedding이 의심되면 `embedding_provider='mock'` row를 확인하고 OpenAI provider로 재실행한다.
