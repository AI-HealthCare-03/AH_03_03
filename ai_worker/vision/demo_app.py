"""
ai_worker/vision/demo_app.py

GPT Vision + OCR 통합 단독 실행용 FastAPI 앱.
팀 전체 서버와 별개로 로컬에서 단독 테스트할 수 있습니다.

실행 방법:
    # .venv-ocr 가상환경 활성화 상태에서
    python -m uvicorn ai_worker.vision.demo_app:app --reload --port 8001

접속:
    GPT Vision 데모 → http://localhost:8001/
    OCR 데모       → http://localhost:8001/ocr
    Swagger UI     → http://localhost:8001/docs
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .ocr.router import router as ocr_router
from .router import router as vision_router

app = FastAPI(
    title="건강 AI 분석 데모",
    description="GPT Vision · PaddleOCR 기반 이미지 분석 API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(vision_router)
app.include_router(ocr_router)


@app.get("/health", tags=["운영"])
async def health_check():
    return {"status": "ok", "modules": ["vision", "ocr"]}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_vision_demo():
    """GPT Vision 데모 페이지."""
    html_path = Path(__file__).parent / "demo.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h2>demo.html 파일을 찾을 수 없습니다.</h2>")


@app.get("/ocr", response_class=HTMLResponse, include_in_schema=False)
async def serve_ocr_demo():
    """OCR 데모 페이지."""
    html_path = Path(__file__).parent / "ocr" / "demo.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h2>ocr/demo.html 파일을 찾을 수 없습니다.</h2>")
