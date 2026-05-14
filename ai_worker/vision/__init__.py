"""
ai_worker/vision/

GPT Vision 기반 이미지 분석 모듈 (MVP).

기능:
    - 식단 이미지 → 음식명 / 카테고리 / 신뢰도 추출
    - 처방전 / 약봉투 → 약품명 / 용량 / 수량 추출
    - 건강검진표 → 4대 만성질환 관련 수치 추출

도메인 연동:
    - DIET  : 식단 분석 결과 수신 및 영양성분 계산
    - MED   : 처방전 분석 결과 수신
    - HEALTH: 건강검진 수치 수신

추후 확장:
    - ai_worker/llm/ : 분석 결과 기반 건강 코멘트 생성
    - ai_worker/rag/ : 의학 문서 검색 후 코멘트에 반영
"""

from .client import AnalysisType, VisionClient
from .router import router
from .schemas import (
    CheckupAnalysisResponse,
    DietAnalysisResponse,
    PrescriptionAnalysisResponse,
)

__all__ = [
    "router",
    "VisionClient",
    "AnalysisType",
    "DietAnalysisResponse",
    "PrescriptionAnalysisResponse",
    "CheckupAnalysisResponse",
]