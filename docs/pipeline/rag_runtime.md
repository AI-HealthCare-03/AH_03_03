# AI Worker RAG

실제 RAG 처리 로직의 책임 영역은 `ai_runtime/llm/rag/`입니다.

- 임베딩 생성
- 벡터 검색 또는 키워드 검색
- retrieval ranking
- LLM context 구성
- RAG 기반 응답 생성

FastAPI `app/services/rag.py`는 RAG 소스/문서/청크/검색 로그의 DB 메타데이터 CRUD까지만 담당합니다.
실제 검색 및 생성 로직은 이 폴더 하위에서 별도 구현합니다.
