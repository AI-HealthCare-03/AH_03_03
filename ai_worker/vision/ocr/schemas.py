"""
ai_worker/vision/ocr/schemas.py
건강검진표 OCR 응답 스키마 (MVP).
"""

from enum import Enum

from pydantic import BaseModel, Field


class ImageQualityStatus(str, Enum):
    GOOD    = "good"
    BLURRY  = "blurry"
    DARK    = "dark"
    SKEWED  = "skewed"
    SMALL   = "small"


QUALITY_GUIDE: dict[str, dict] = {
    "blurry": {
        "message": "사진이 흐립니다.",
        "guide": [
            "카메라를 고정하고 흔들리지 않게 찍어주세요.",
            "검진표와 카메라 사이 거리를 30~50cm로 유지해주세요.",
            "자동 초점이 맞춰진 후 촬영해주세요.",
        ],
    },
    "dark": {
        "message": "사진이 너무 어둡습니다.",
        "guide": [
            "밝은 곳에서 촬영해주세요.",
            "형광등 아래나 창가에서 촬영하면 좋습니다.",
            "플래시를 사용하면 반사가 생길 수 있으니 주의하세요.",
        ],
    },
    "skewed": {
        "message": "검진표가 기울어져 있습니다.",
        "guide": [
            "검진표를 평평한 곳에 놓고 촬영해주세요.",
            "카메라를 검진표와 수직(정면)이 되도록 맞춰주세요.",
            "검진표 네 모서리가 모두 보이도록 찍어주세요.",
        ],
    },
    "small": {
        "message": "이미지 해상도가 낮습니다.",
        "guide": [
            "더 가까이서 촬영하거나 고해상도로 설정 후 찍어주세요.",
            "검진표 전체가 화면을 꽉 채우도록 찍어주세요.",
        ],
    },
}


class OcrStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED  = "failed"


STATUS_MESSAGE: dict[str, str] = {
    "success": "검진표 수치 추출이 완료되었습니다.",
    "partial": "일부 항목만 인식되었습니다. 누락된 항목을 직접 입력해주세요.",
    "failed":  "수치를 인식하지 못했습니다. 다시 촬영하거나 직접 입력해주세요.",
}


class CheckupOcrData(BaseModel):
    """4대 만성질환(고혈압·당뇨·이상지질혈증·비만) 관련 수치."""
    systolic_bp:       float | None = Field(None, description="수축기 혈압 (mmHg)")
    diastolic_bp:      float | None = Field(None, description="이완기 혈압 (mmHg)")
    fasting_glucose:   float | None = Field(None, description="공복혈당 (mg/dL)")
    hba1c:             float | None = Field(None, description="당화혈색소 (%)")
    total_cholesterol: float | None = Field(None, description="총콜레스테롤 (mg/dL)")
    triglyceride:      float | None = Field(None, description="중성지방 (mg/dL)")
    hdl:               float | None = Field(None, description="HDL (mg/dL)")
    ldl:               float | None = Field(None, description="LDL (mg/dL)")
    height_cm:         float | None = Field(None, description="키 (cm)")
    weight_kg:         float | None = Field(None, description="몸무게 (kg)")
    bmi:               float | None = Field(None, description="BMI")
    waist_cm:          float | None = Field(None, description="허리둘레 (cm)")


class ImageQualityReport(BaseModel):
    status:     ImageQualityStatus
    message:    str
    guide:      list[str] = Field(default_factory=list)
    blur_score: float | None = None
    brightness: float | None = None
    skew_angle: float | None = None


class CheckupOcrResponse(BaseModel):
    ocr_status:            OcrStatus
    message:               str
    quality_report:        ImageQualityReport
    extracted_data:        CheckupOcrData = Field(default_factory=CheckupOcrData)
    low_confidence_fields: list[str]      = Field(default_factory=list)
    raw_text:              list[str]      = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error_code: str
    message:    str


ERROR_MAP: dict[str, dict] = {
    "image_too_large":  {"status_code": 413, "message": "이미지 크기가 너무 큽니다. 20MB 이하로 다시 시도해주세요."},
    "image_too_small":  {"status_code": 422, "message": "이미지가 손상되었거나 너무 작습니다. 다시 촬영해주세요."},
    "unsupported_type": {"status_code": 415, "message": "지원하지 않는 형식입니다. JPG, PNG, WEBP로 업로드해주세요."},
    "quality_failed":   {"status_code": 422, "message": "이미지 품질이 낮아 분석할 수 없습니다. 안내에 따라 다시 촬영해주세요."},
    "ocr_error":        {"status_code": 500, "message": "OCR 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."},
    "server_error":     {"status_code": 500, "message": "서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."},
}
