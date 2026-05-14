"""
ai_worker/vision/demo_app.py

GPT Vision 모듈 단독 실행용 FastAPI 앱.
팀 전체 서버와 별개로 로컬에서 단독 테스트할 수 있습니다.

실행 방법:
    uvicorn ai_worker.vision.demo_app:app --reload --port 8001

접속:
    데모 페이지 → http://localhost:8001/
    Swagger UI → http://localhost:8001/docs
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .router import router

app = FastAPI(
    title="GPT Vision 분석 데모",
    description="식단 · 처방전 · 건강검진표 이미지 분석 API (gpt-4o-mini)",
    version="0.1.0",
)

# CORS 설정 (데모용 — 운영 시 팀 도메인으로 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vision 라우터 등록
app.include_router(router)


@app.get("/health", tags=["운영"])
async def health_check():
    """서버 상태 확인."""
    return {"status": "ok", "module": "vision"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_demo():
    """데모 프론트 페이지 제공."""
    html_path = Path(__file__).parent / "demo.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h2>demo.html 파일을 이 디렉토리에 넣어주세요.</h2>")