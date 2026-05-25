FROM python:3.13-slim

WORKDIR /app

# 1. uv 설치 (공식 이미지에서 복사해오는 가장 깔끔한 방법)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 2. 의존성 패키지 관리 파일을 컨테이너 내부에 복사합니다.
COPY pyproject.toml uv.lock ./

# 3. Redis Stream consumer가 DB 상태를 갱신하므로 app + ai 그룹을 함께 설치합니다.
RUN uv sync --group app --group ai --no-dev --frozen

# 4. 로컬 소스 코드를 컨테이너 내부로 복사합니다.
COPY ./ai_runtime ./ai_runtime
COPY ./app ./app

# 5. 실제 AI Worker 진입점입니다. 현재는 DEMO_ECHO Redis Stream job만 처리합니다.
CMD ["uv", "run", "--no-sync", "python", "-m", "ai_runtime.main"]
