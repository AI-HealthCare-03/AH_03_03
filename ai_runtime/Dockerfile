FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*


# 1. uv 설치 (공식 이미지에서 복사해오는 가장 깔끔한 방법)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 2. 의존성 패키지 관리 파일을 컨테이너 내부에 복사합니다.
COPY pyproject.toml uv.lock ./

# 3. Redis Stream consumer가 DB 상태를 갱신하고 OCR을 실행하므로 app + ai + ocr 그룹을 함께 설치합니다.
RUN uv sync --group app --group ai --group ocr --no-dev --frozen

# 4. 로컬 소스 코드를 컨테이너 내부로 복사합니다.
COPY ./ai_runtime ./ai_runtime
COPY ./app ./app
COPY ./docs/rag_sources ./docs/rag_sources

# 5. 실제 AI Worker 진입점입니다. Redis Stream 기반 job consumer를 실행합니다.
CMD ["uv", "run", "--no-sync", "python", "-m", "ai_runtime.main"]
