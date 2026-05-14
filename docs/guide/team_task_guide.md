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

## 팀원별 담당 영역

Backend API 담당:

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

CV 담당:

- 이미지 분석
- 식단 OCR 또는 분류 실험
- MVP 포함 여부에 따른 독립 실험 코드 관리

LLM/RAG 담당:

- LLM 호출 방식 검토
- 프롬프트 설계
- 문서 chunking, embedding, retrieval 실험
- MVP 범위 밖 기능은 실서비스 DB와 분리해서 관리

Pipeline(조장이 주말에 좀 작업좀 할께요)

- 건강위험 분석 전체 흐름 정리
- ML 결과, SHAP factor, 챌린지 추천 연결
- Backend와 AI Worker 사이 입출력 형식 조율

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

Docker Compose로 전체 실행:

```bash
docker compose up -d --build
```

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
