# 팀 작업 가이드

## 브랜치 전략

- `main`: 안정/배포 기준 브랜치
- `develop`: 공용 개발 브랜치
- `feature/*`: 개인 기능 개발 브랜치

기능 개발은 `feature/*` 브랜치에서 진행하고, PR을 통해 `develop`에 합친다. `main`은 배포 또는 안정화 기준으로 관리한다.

## 공용 설정 파일 수정 금지 규칙

아래 파일은 수정발생시 알려주세요

- `pyproject.toml`
- `uv.lock`
- `docker-compose.yml`
- `.github/workflows/*`
- `app/core/*`
- `envs/example.local.env`
- `envs/example.prod.env`
- `README.md`

공용 설정 파일은 한 명의 변경이 전체 팀 실행 환경과 CI에 영향을 줄 수 있으므로, 수정 전 반드시 목적과 영향 범위를 공유한다.

## env 관리 규칙

- 실제 `.env` 파일은 push하지 않는다.
- `envs/example.local.env`를 복사해서 `envs/.local.env`로 사용한다.
- Docker Compose 기준 DB 접속은 `DB_HOST=postgres`를 사용한다.
- FastAPI를 로컬에서 직접 실행하는 기준은 `DB_HOST=localhost`를 사용한다.
- 개인 로컬 비밀번호, API key, 토큰은 Git에 올리지 않는다.

## 작업 영역 분리 기준

- Backend API: `app/`
- ML: `ai_worker/ml/`
- CV: `ai_worker/cv/`
- LLM/RAG: `ai_worker/llm/`, `ai_worker/rag/`
- Pipeline: `ai_worker/pipelines/`

ML/CV/LLM/RAG 담당자는 `app/`을 직접 크게 수정하지 않는다. 백엔드 담당자는 `ai_worker` 내부 모델 코드를 직접 크게 수정하지 않는다. API와 Worker 사이 데이터 형식은 DTO/schema 기준으로 합의 후 수정한다.

## 풀서비스 요구사항 정리 기준

풀서비스 1차 범위는 `docs/design/full_service_scope.md`를 기준으로 관리한다.
코치님/조교님 피드백 반영 기준은 `docs/design/requirements_refactor_notes.md`를 따른다.

- 요구사항 정의서는 사용자 관점 기능 중심으로 정리한다.
- NFR, 아키텍처, 재시도, 공통 컴포넌트, worker/queue 같은 구현 세부사항은 별도 설계 문서로 분리한다.
- 회원가입은 기본정보 중심으로 유지하고, 건강 수치와 모델 입력값은 가입 후 건강정보 입력 화면에서 받는다.
- 이메일 인증은 유지한다.
- 소셜 로그인과 웨어러블 연동은 1차 범위 제외/보류로 둔다.
- 휴대폰 SMS 인증은 Twilio Verify 기반 구현 대상으로 유지한다.
- 휴대폰 번호 중복확인은 유지 가능하다.
- RAG, ML, CV, 알림, 가족, 관리자, 모니터링은 풀서비스 구현 대상으로 유지하되 후속 로드맵으로 관리한다.

## 팀원별 담당 영역

FastAPI 담당:

- 인증/사용자 API
- 건강정보 API
- 분석 요청/결과 조회 API
- 챌린지 API
- Dashboard API
- Notification API

ML 담당:

- 만성질환 위험도 예측
- tabular feature 전처리
- SHAP 또는 feature contribution 계산
- recall, F1, confusion matrix 중심 평가
- model version, threshold, feature schema 관리

CV 담당:

- 이미지 분석
- 식단 OCR 또는 분류 실험
- 건강검진표 OCR
- 복약/처방전 OCR
- 식단 이미지 기반 간편 분석과 사용자 보정 흐름

LLM/RAG 담당:

- LLM 호출 방식 검토
- 프롬프트 설계
- 문서 chunking, embedding, retrieval 실험
- LangChain/LangGraph, embedding model, pgvector/FAISS/Chroma 후보 검토
- RAGAS/LangSmith/Langfuse 평가/관측 전략 검토

Pipeline(조장이 주말에 좀 작업좀 할께요)

- 건강위험 분석 전체 흐름 정리
- ML 결과, SHAP factor, 챌린지 추천 연결
- Backend와 AI Worker 사이 입출력 형식 조율

Frontend 담당:

- 회원가입/로그인 UX
- 건강정보 입력 UX
- 분석 결과/대시보드
- 챌린지/식단/복약/알림 화면
- 접근성/반응형 고도화

먼저 완료한 인원은 FastAPI 또는 Frontend 병목 작업을 지원한다.

## MVP 포함 범위

- AUTH / USER
- MAIN / MYPAGE
- HEALTH
- ANALYSIS + SHAP
- CHALLENGE
- DASHBOARD 조회/집계
- NOTIFICATION 최소기능

## MVP 제외 범위

- DIET 실서비스 DB
- LLM 실서비스 DB
- FAMILY
- QNA
- MEDICATION
- ADMIN
- 운영 로그 상세
- 외부 알림 SMS/Email/Push/Kakao

## PR/커밋 규칙

PR 규칙:

- PR은 가능한 작은 단위로 올린다.
- 공용 설정 파일 변경이 있으면 PR 설명에 반드시 표시한다.
- DB 모델 변경이 있으면 migration 필요 여부를 명시한다.
- API 응답 형식 변경이 있으면 프론트/AI 담당자에게 공유한다.
- CI 실패 상태로 merge하지 않는다.

커밋 컨벤션:

- `✨ feat`: 기능 추가
- `🐛 fix`: 버그 수정
- `💡 chore`: 기능 변경 없는 작업
- `📝 docs`: 문서 수정
- `🚚 build`: 빌드/환경 설정
- `✅ test`: 테스트
- `♻️ refactor`: 리팩터링

## 개발 실행 방법

의존성 설치:

```bash
uv sync --group app --group dev
```

Docker Compose로 기존 루트 스택 실행:

```bash
docker compose up -d --build
```

개발 서버용 Docker 전체 스택 실행:

```bash
docker network create ai-health-shared
docker compose -f infra/docker/docker-compose.dev.yml up -d --build
DB_HOST=localhost uv run python scripts/setup_local_mvp_db.py
```

접속:

- 웹/Nginx: `http://localhost:8080`
- API Docs: `http://localhost:8080/api/docs`
- FastAPI 직접 접근: `http://localhost:8000`

Docker 개발 스택은 `frontend`, `nginx`, `fastapi`, `ai-worker`, `postgres`, `redis`를 포함한다.
`frontend`는 Docker build 단계에서 정적 파일을 만들고 내부 Nginx로 서빙하며, 바깥 `nginx`가 `/`를 frontend로, `/api/`를 FastAPI로 proxy한다.

Langfuse는 `infra/langfuse/docker-compose.yml`로 별도 실행한다. 우리 서비스와 Langfuse를 같은 Docker 서버에서 연결해야 하면 `ai-health-shared` external network를 공유한다. 이 네트워크는 HTTP 접근용이며, 우리 서비스 Postgres/Redis와 Langfuse Postgres/Redis는 공유하지 않는다.

```bash
cd infra/langfuse
cp .env.example .env
docker compose up -d
```

Docker network 내부에서 Langfuse를 볼 때는 `LANGFUSE_HOST=http://langfuse-web:3000`, 호스트에서 직접 접근할 때는 `LANGFUSE_HOST=http://localhost:3000`을 사용한다. Langfuse SDK 연동은 후속 작업이다.

FastAPI 로컬 직접 실행:

```bash
uv run uvicorn app.main:app --reload
```

테스트:

```bash
uv run pytest app
```

Lint:

```bash
uv run ruff check .
uv run ruff format . --check
```

## 웹 MVP 로컬 시연 흐름

현재 `feature/kdu-web`의 웹 MVP는 FastAPI가 건강 분석, OCR, 챗봇 응답을 동기 더미 로직으로 처리한다. 실제 ML/CV/LLM 모델은 아직 붙이지 않았고, 프론트 화면 흐름과 API 계약을 확인하기 위한 시연용 구조다.

`Architecture_ver1.drawio`의 `async_jobs`, Redis Stream, AI Worker, SSE, Notification Worker, Report Worker는 후속 모델 연동/운영 단계에서 도입한다. 지금 로컬 MVP 테스트에서는 PostgreSQL, FastAPI, React/Vite만 먼저 띄우면 된다.

로컬 실행 순서:

```bash
docker compose up -d postgres
DB_HOST=localhost uv run python scripts/setup_local_mvp_db.py
DB_HOST=localhost uv run uvicorn app.main:app --reload
cd frontend
npm install
npm run dev
```

데모 계정:

- `demo@example.com` / `Demo1234!`
- `demo_high@example.com` / `Demo1234!`

관리자 콘솔 로컬 계정:

- `admin@example.com` / `Demo1234!` (`SUPER_ADMIN`)
- `monitor@example.com` / `Demo1234!` (`MONITOR`)

관리자 콘솔 진입:

1. `DB_HOST=localhost uv run python scripts/setup_local_mvp_db.py`로 로컬 seed를 실행한다.
2. 관리자 계정으로 로그인한다.
3. 좌측 사이드바의 “관리자 콘솔”을 클릭하거나 `/admin`에 직접 접속한다.
4. `MONITOR` 계정은 시스템 상태와 오류 로그 조회 중심이며, 민감정보 접근 로그는 `ADMIN` 이상 정책에 따라 제한될 수 있다.

Seed 실행 내용:

- `scripts/init_local_dev_db.py`: 로컬 MVP용 Tortoise schema 생성
- `scripts/seed_mvp_challenges.py`: 챌린지 마스터 생성
- `scripts/seed_mvp_faqs.py`: FAQ 생성
- `scripts/seed_demo_users.py`: 데모 사용자와 건강정보/분석/챌린지/식단/복약/알림 데이터, 로컬 관리자 콘솔 계정 생성
- `scripts/setup_local_mvp_db.py`: 위 흐름 통합 실행

주의:

- 이 흐름은 로컬 MVP 테스트 전용이다.
- Aerich migration에 old format 이슈가 있으므로 빈 로컬 DB는 `setup_local_mvp_db.py`로 보강한다.
- 운영/공유 DB는 migration 파일 정리 후 Aerich 기준으로 적용한다.
- 루트 `.env`와 `frontend/.env`는 구분해서 관리한다.
- Firebase Auth와 소셜 로그인은 1차 풀서비스 범위에서 제외/보류한다. 기본 인증은 FastAPI JWT Auth와 이메일 인증이다.
