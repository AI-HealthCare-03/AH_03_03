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

```env
RAG_EMBEDDING_ENABLED=true
RAG_EMBEDDING_PROVIDER=openai
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_DIMENSION=1536
RAG_EMBEDDING_BATCH_SIZE=64
OPENAI_API_KEY=<openai-api-key>
```

Langfuse trace를 확인하려면 Langfuse 설정도 필요하다.

```env
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=<langfuse-url>
LANGFUSE_PUBLIC_KEY=<langfuse-public-key>
LANGFUSE_SECRET_KEY=<langfuse-secret-key>
```

## Dry-run

먼저 DB write 없이 대상 chunk와 예상 작업량을 확인한다.

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

```bash
uv run python scripts/rag/embed_rag_chunks.py --provider openai --apply --json
```

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
- 비용과 rate limit을 고려해 batch size를 조절한다.
- OpenAI key가 없으면 OpenAI provider 생성은 실패해야 한다.
- pgvector index는 별도 단계다. chunk 수가 커지면 vector search 성능을 재검토한다.
- vector/hybrid retriever를 서비스 API에 연결하는 작업은 embedding 저장과 검색 품질 확인 이후에 진행한다.

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

## Rollback/복구

embedding 저장은 RAG chunk 원문과 metadata를 바꾸지 않는다. 문제가 있으면 같은 script를 올바른 provider/model/dimension으로 다시 실행해 재생성한다.

운영에서 잘못 저장된 `mock` embedding이 의심되면 `embedding_provider='mock'` row를 확인하고 OpenAI provider로 재실행한다.
