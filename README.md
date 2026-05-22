# AI Healthcare Project Template

이 프로젝트는 AI 모델 추론(Inference) 워커와 FastAPI API 서버를 통합한 서비스 템플릿입니다. 
현대적인 Python 패키지 관리 도구인 `uv`와 컨테이너화 도구인 `Docker`를 활용하여 일관된 개발 및 배포 환경을 제공합니다.

---

## 🚀 주요 특징

- **FastAPI Framework**: 고성능 비동기 API 서버 구현.
- **AI Worker**: 모델 추론 및 학습 작업을 API 서버와 분리하여 처리.
- **UV Package Manager**: 매우 빠른 의존성 설치 및 가상환경 관리.
- **Tortoise ORM**: 비동기 방식의 데이터베이스 모델링 및 쿼리 관리.
- **Docker-Compose**: PostgreSQL(pgvector), Redis, Nginx를 포함한 전체 서비스 스택을 한 번에 실행.
- **CI/CD Scripts**: 코드 포맷팅(Ruff), 타입 체크(Mypy), 테스트(Pytest)를 위한 자동화 스크립트 제공.

---

## 📂 프로젝트 구조

```text
.
├── ai_worker/          # AI 모델 추론 및 학습 관련 코드 (Worker)
│   ├── core/           # 워커 설정 및 로거
│   ├── models/         # AI 모델 파일 보관 (PyTorch 등)
│   ├── tasks/          # 실제 처리할 작업 정의
│   └── main.py         # 워커 진입점
├── app/                # FastAPI 서버 코드
│   ├── apis/           # API 라우터 (v1 버전 관리)
│   ├── core/           # 서버 설정 (pydantic-settings), DB 설정, JWT, Validator 등 핵심 기능
│   ├── dtos/           # 데이터 전송 객체 (Pydantic models)
│   ├── models/         # DB 테이블 정의
│   ├── services/       # 비즈니스 로직
│   └── main.py         # FastAPI 애플리케이션 진입점
├── envs/               # 환경 변수 설정 파일 (.env)
├── infra/              # 인프라 설정 관련 디렉터리
│   ├── docker/         # Docker Compose 설정 (운영용)
│   └── nginx/          # Nginx 설정 파일 (리버스 프록시)
├── scripts/            # 배포 및 CI용 쉘 스크립트
├── docker-compose.yml  # 로컬 개발용 서비스 실행 설정
└── pyproject.toml      # uv 기반 의존성 관리 설정
```

---

## ⚙️ 사전 준비 사항

- **Python**: 3.13 이상 (로컬 개발 환경용)
- **UV**: Python 패키지 매니저 ([설치 가이드](https://github.com/astral-sh/uv))
- **Docker & Docker-Compose**: 전체 서비스 실행용

---

## 🛠️ 설치 및 설정

### 1. 가상환경 구축 및 의존성 설치

`uv`를 사용하여 프로젝트에 필요한 패키지를 설치합니다.

```bash
# 의존성 설치 (가상환경 자동 생성)
uv sync

# 특정 그룹의 의존성만 설치하려는 경우
uv sync --group app  # API 서버용
uv sync --group ai   # AI 워커용
```

### 2. 환경 변수 설정

`envs/` 디렉토리에 있는 예시 파일을 복사하여 `.env` 파일을 생성합니다.
- 로컬용 
    ```bash
    cp envs/example.local.env envs/.local.env
    ```
- 배포용 
    ```bash
    cp envs/example.prod.env envs/.prod.env
    ```

생성된 `env` 파일 내의 환경변수들은 프로젝트 상황에 맞게 수정하세요.
Docker Compose 내부에서 API 서버를 실행하는 경우 `DB_HOST=postgres`를 사용하고, FastAPI를 로컬에서 직접 실행하면서 DB만 Docker로 띄우는 경우 `DB_HOST=localhost`를 사용하세요.

---

## 🏃 실행 방법

### 0. 웹 MVP 시연용 빠른 실행

현재 웹 MVP는 **FastAPI에서 건강 분석, OCR, 챗봇 응답을 동기 더미 로직으로 처리하는 시연용 구조**입니다.
실제 ML/CV/LLM 모델 호출은 아직 연결하지 않았고, 프론트 화면 흐름과 API 계약을 확인하기 위한 단순 응답을 사용합니다.

> `Architecture_ver1.drawio`에 정리된 `async_jobs`, Redis Stream, AI Worker, SSE, Notification Worker, Report Worker 구조는
> 후속 모델 연동/운영 단계에서 도입할 예정입니다. 현재 로컬 MVP 실행에는 필수 구성요소가 아닙니다.

로컬에서 프론트와 백엔드를 같이 확인할 때는 아래 순서로 실행합니다.

```bash
# 1. PostgreSQL(pgvector) 컨테이너 실행
docker compose up -d postgres

# 2. 로컬 MVP 테스트용 테이블 생성 및 seed 실행
DB_HOST=localhost uv run python scripts/setup_local_mvp_db.py

# 3. FastAPI 실행
DB_HOST=localhost uv run uvicorn app.main:app --reload

# 4. React/Vite 프론트 실행
cd frontend
npm install
npm run dev
```

접속 주소:

- FastAPI Swagger: `http://localhost:8000/docs`
- FastAPI healthcheck: `http://localhost:8000/api/v1/system/health`
- React/Vite: `http://localhost:5173`

데모 계정:

- `demo@example.com` / `Demo1234!`
- `demo_high@example.com` / `Demo1234!`

Seed 포함 데이터:

- FAQ
- 챌린지 마스터
- 데모 사용자
- 건강정보, 분석 결과, 챌린지 참여/로그, 식단, 복약, 알림, 검진표 OCR 시연 데이터

주의:

- `scripts/setup_local_mvp_db.py`는 로컬 MVP 테스트 전용입니다.
- Aerich migration에 old format 이슈가 남아 있어 로컬 빈 DB는 `setup_local_mvp_db.py`로 보강합니다.
- 운영/공유 DB는 migration 파일 정리 후 Aerich 기준으로 적용해야 합니다.
- 루트 `.env`는 백엔드/공용 환경변수용이고, `frontend/.env`는 Vite 프론트용입니다.
- Firebase Auth와 소셜 로그인은 1차 풀서비스 범위에서 제외/보류합니다. 현재 로그인/회원가입은 FastAPI JWT Auth와 이메일 인증 기준이며, Firebase 전용 사용자 컬럼은 사용하지 않습니다.
- 향후 소셜 로그인을 재도입할 경우 `users` 테이블에 provider 정보를 직접 넣지 않고 별도 OAuth 계정 연결 테이블을 설계합니다.
- 로컬 회원가입 이메일 인증은 `EMAIL_ENABLED=false`일 때 실제 SMTP 발송 대신 `/api/v1/auth/email-verifications/send` 응답의 `debug_code`로 확인합니다. 프론트는 로컬 테스트 편의를 위해 이 값을 인증코드 입력칸에 자동 반영합니다.
- 운영환경에서는 이메일 인증 `debug_code`와 비밀번호 재설정 `debug_token`을 응답하지 않습니다. `EMAIL_ENABLED=true`와 SMTP 환경변수를 설정하면 실제 인증코드/비밀번호 재설정 링크를 발송합니다.
- 휴대폰 SMS 인증은 Twilio Verify 기반입니다. 서버는 한국 휴대폰 번호를 Twilio 호출용 E.164 형식(`+821012345678`)으로 정규화하고, 기존 사용자 DB 호환을 위해 저장/중복확인은 로컬 번호(`01012345678`) 기준도 함께 확인합니다. 로컬에서는 `TWILIO_ENABLED=false`로 개발용 인증번호 흐름을 사용할 수 있지만, 운영환경에서는 debug code를 응답하지 않으며 `TWILIO_ENABLED=true`와 Twilio Verify secret 설정이 필요합니다.
- 휴대폰 인증번호 요청은 같은 번호 기준 60초 이내 재발송을 제한하고, 1시간 5회 초과 요청을 제한합니다. 인증번호 확인 실패도 같은 번호 기준 5회 이상이면 일정 시간 제한합니다.
- 회원가입 주소는 초기 서비스 지역 확인용 필수 기본정보입니다. 상세주소 수집 범위와 보관 정책은 운영 전 개인정보 최소수집 원칙에 따라 재검토합니다.
- 풀서비스 관리자 권한은 `users.role`을 기준으로 판단합니다. `is_admin`은 legacy 호환 필드로만 남기고, 관리자 role은 `USER/MONITOR/OPERATOR/ADMIN/SUPER_ADMIN` 구조로 설계합니다. 운영 단계에서는 관리자 서브도메인, 2FA, audit log를 추가해야 합니다.
- 로그인 실패가 5회 이상 누적되면 CAPTCHA 등 추가 확인을 요구하는 soft-lock 정책을 적용합니다. CAPTCHA 도입 전에는 짧은 제한과 일반화된 안내 메시지를 사용합니다. `/api/v1/system/health`는 DB/Redis 상태를 포함하고, 모든 응답은 `X-Request-ID` 헤더로 요청 추적값을 반환합니다.
- 로그인 시각은 `last_login_at`을 표준 필드로 사용합니다. `last_login` legacy 컬럼은 제거 대상이며 신규 코드에서 사용하지 않습니다.
- 모든 응답은 `X-Request-ID`를 포함합니다. 처리되지 않은 500 서버 예외는 `system_error_logs`에 최소 추적 정보만 저장하며, request body와 민감정보 원문은 저장하지 않습니다.
- 건강정보/분석결과/검진표/복약정보/대시보드 조회는 `sensitive_access_logs`에 접근 사실을 남깁니다. 건강 수치 원문, 토큰, 인증코드, request body는 저장하지 않습니다.
- 비밀번호 해싱은 Argon2id 단일 방식입니다. 이전 로컬 계정의 예전 해시는 호환하지 않으므로 로그인되지 않으면 재가입하거나 비밀번호 재설정을 진행하세요. 운영 전환 시에는 별도 재설정/전환 정책이 필요합니다.
- AI Worker, `async_jobs`, Redis queue 기반 비동기 모델 처리 연결은 후속 ML/CV/LLM 운영 연동 단계에서 진행합니다.
- 풀서비스 1차 범위는 [Full Service Scope](docs/design/full_service_scope.md)를 기준으로 관리합니다. 1차 제외/보류 항목은 소셜 로그인, 웨어러블 연동 2개이며, 휴대폰 SMS 인증은 Twilio Verify 기반 구현 대상으로 유지합니다.
- 코치님/조교님 피드백 반영 기준은 [Requirements Refactor Notes](docs/design/requirements_refactor_notes.md)에 정리합니다. 요구사항 정의서는 사용자 기능 중심으로 유지하고, NFR/아키텍처/재시도/공통 컴포넌트는 별도 설계 문서로 분리합니다.

### 1. 로컬 및 개발 환경

#### Docker Compose로 전체 스택 실행

모든 서비스(API, Worker, DB, Redis, Nginx)를 한 번에 실행합니다.

```bash
docker-compose up -d --build
```

실행 후 다음 주소로 접속 가능합니다:
- **API 서버**: [http://localhost/api/docs](http://localhost/api/docs) (Swagger UI)
- **Nginx**: 80 포트를 통해 API 서버로 요청을 전달합니다.

DB는 PostgreSQL(pgvector) 컨테이너로 실행됩니다. 기존 MySQL 기준 Aerich migration을 사용하던 경우에는 migration 파일을 임의로 삭제하지 말고, 팀에서 PostgreSQL 기준으로 재초기화할지 먼저 결정한 뒤 `aerich init-db` 또는 신규 migration 생성 절차를 진행하세요.

모델 변경 후 migration을 생성/적용할 때는 app dependency group을 포함해서 실행합니다.

```bash
uv run --group app aerich migrate --name <migration_name>
uv run --group app aerich upgrade
```

#### 로컬에서 개별 실행 (개발용)

**FastAPI 서버 실행:**
```bash
DB_HOST=localhost uv run uvicorn app.main:app --reload
# or
docker compose up -d --build app
```

**AI Worker 실행:**
```bash
uv run python -m ai_worker.main
# or
docker compose up -d --build ai_worker
```

**로컬 MVP 데모 DB 준비:**
프론트 MVP 화면을 빈 데이터 없이 확인하려면 PostgreSQL 컨테이너를 띄운 뒤 로컬 전용 seed를 실행합니다.
이 방식은 로컬 MVP 테스트용이며, 운영/공유 DB에서는 Aerich migration 기준으로 관리해야 합니다.

```bash
docker compose up -d postgres
DB_HOST=localhost uv run python scripts/setup_local_mvp_db.py
```

생성되는 데모 계정:
- `demo@example.com` / `Demo1234!`
- `demo_high@example.com` / `Demo1234!`

### 2. EC2 배포 환경 (Production)

제공된 쉘 스크립트를 사용하여 AWS EC2 환경에 이미지를 빌드, 푸시 및 배포할 수 있습니다.

#### 사전 준비
- EC2 인스턴스 (Ubuntu 권장)
- SSH 키 페어 (`~/.ssh/` 경로에 위치)
- 도커 허브(Docker Hub) 계정 및 Personal Access Token
- 배포용 환경 변수 설정 (`envs/.prod.env`)
- 도메인 구매 (Gabia, GoDaddy, AWS Route53 등)

#### 자동 배포 스크립트 실행
`scripts/deployment.sh`는 도커 이미지 빌드, 레포지토리 푸시, EC2 접속 및 컨테이너 실행 과정을 자동화합니다.

```bash
chmod +x scripts/deployment.sh
./scripts/deployment.sh
```
스크립트 실행 시 다음 정보를 입력해야 합니다:
1. 도커 허브 계정 정보 (Username, PAT)
2. 이미지를 업로드할 레포지토리 이름
3. 배포할 서비스 선택 (FastAPI, AI-Worker) 및 버전(Tag)
4. SSH 키 파일명 및 EC2 IP 주소
5. https 사용여부
   - 5-1. https인 경우 도메인 추가 입력  

#### SSL(HTTPS) 설정 (Certbot)
도메인을 연결하고 HTTPS를 적용하려면 `scripts/certbot.sh`를 사용합니다.

```bash
chmod +x scripts/certbot.sh
./scripts/certbot.sh
```
1. 도메인 주소 및 이메일 입력
2. SSH 키 파일명 및 EC2 IP 주소 입력
3. Let's Encrypt를 통한 인증서 발급 및 Nginx 설정 자동 갱신 적용

---

## 🧪 테스트 및 품질 관리

제공된 스크립트를 사용하여 코드의 품질을 검증할 수 있습니다.

```bash
# 테스트 실행
./scripts/ci/run_test.sh

# 코드 포맷팅 확인 (Ruff)
./scripts/ci/code_fommatting.sh

# 정적 타입 검사 (Mypy)
./scripts/ci/check_mypy.sh
```

---

## 📝 개발 가이드

- **API 추가**: `app/apis/v1/` 아래에 새로운 라우터 파일을 생성하고 `app/apis/v1/__init__.py`에 등록하세요.
- **DB 모델 추가**: `app/models/`에 Tortoise 모델을 정의하고 `app/db/databases.py`의 `MODELS` 리스트에 추가하세요.
- **AI 로직 추가**: `ai_worker/tasks/`에 새로운 처리 로직을 작성하고 `ai_worker/main.py`에서 호출하도록 구성하세요.

## 개발 작업 영역 분리 기준

- 백엔드 API 담당자는 `app/` 하위에서 작업합니다.
- ML 담당자는 `ai_worker/ml/` 하위에서 작업합니다.
- CV 담당자는 `ai_worker/cv/` 하위에서 작업합니다.
- LLM 담당자는 `ai_worker/llm/` 하위에서 작업합니다.
- RAG 담당자는 `ai_worker/rag/` 하위에서 작업합니다.
- 여러 AI 처리 흐름을 묶는 파이프라인은 `ai_worker/pipelines/`에서 관리합니다.
- ML/CV/LLM/RAG 담당자는 `app/`을 직접 크게 수정하지 않습니다.
- 백엔드 담당자는 `ai_worker` 내부 모델 코드를 직접 크게 수정하지 않습니다.
- API와 Worker 사이 데이터 형식은 DTO/schema 기준으로 합의 후 수정합니다.
- `pyproject.toml`, `uv.lock`, `docker-compose.yml`, env example, `app/core` 같은 공용 설정 파일은 팀장 승인 없이 수정하지 않습니다.
