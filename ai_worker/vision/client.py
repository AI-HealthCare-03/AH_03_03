"""
ai_worker/vision/client.py

GPT Vision API 호출 클라이언트.
프롬프트 관리 및 이미지 분석 요청을 담당합니다.

MVP 기준으로 작성. 추후 프롬프트 튜닝 및 fallback 로직 보완 예정.
"""

import base64
import json
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ── 분석 유형 ─────────────────────────────────────────────────────────────────

class AnalysisType:
    DIET         = "diet"
    PRESCRIPTION = "prescription"
    CHECKUP      = "checkup"


# ── 프롬프트 ──────────────────────────────────────────────────────────────────
# TODO: 추후 프롬프트 튜닝 예정

PROMPTS: dict[str, str] = {

    AnalysisType.DIET: """
    이 식단 이미지를 분석하세요. 반드시 아래 JSON만 응답하세요 (마크다운 금지):
    {
      "foods": [
        {
          "name": "음식명",
          "nutrient_category": "탄수화물|단백질|지방|식이섬유|비타민|미네랄|항산화|건강식",
          "cooking_method": "조리법 (예: 튀김, 구이, 찜, 볶음, 생것) 또는 null",
          "estimated_amount": "추정 용량 (예: 200g, 1공기, 1개) 또는 null",
          "amount_requires_input": true또는false,
          "nutrition": {
            "칼로리": 숫자또는null,
            "탄수화물": 숫자또는null,
            "당류": 숫자또는null,
            "단백질": 숫자또는null,
            "지방": 숫자또는null,
            "포화지방": 숫자또는null,
            "식이섬유": 숫자또는null,
            "나트륨": 숫자또는null,
            "영양성분_신뢰도": 0.0~1.0,
            "추정값_여부": true
          },
          "confidence": 0.0~1.0,
          "search_keyword": "식약처/영양DB 검색 키워드"
        }
      ],
      "analysis_status": "success|low_confidence|failed",
      "fail_reason": null
    }

    규칙:
    - nutrient_category는 해당 음식의 주요 영양소 기준으로 분류
      탄수화물 → 밥, 빵, 면류, 떡
      단백질   → 닭가슴살, 두부, 생선, 계란
      지방     → 삼겹살, 튀김류, 버터
      식이섬유 → 고구마, 양배추, 당근
      비타민   → 브로콜리, 파프리카, 시금치
      미네랄   → 시금치, 견과류, 콩류
      항산화   → 토마토, 블루베리, 당근
      건강식   → 아보카도, 연어, 올리브유

    - 같은 재료라도 조리법이 다르면 반드시 구분
      예시) 닭고기 → 치킨(튀김) / 닭가슴살(구이/찜) 구분
      예시) 돼지고기 → 삼겹살(구이) / 제육볶음 / 돈까스(튀김) 구분

    - estimated_amount는 이미지에서 용량 추정이 가능하면 작성, 불가능하면 null
    - amount_requires_input은 용량 추정 불가능하면 true
    - nutrition은 estimated_amount 기준으로 추정, 추정 불가 항목은 null

    - 4대 만성질환 관련 영양소 반드시 포함
      고혈압       → 나트륨, 식이섬유
      당뇨         → 당류, 탄수화물, 식이섬유
      이상지질혈증  → 포화지방, 식이섬유
      비만         → 칼로리, 지방

    - 영양성분_신뢰도는 추정 근거에 따라 유동적으로 설정
      음식명이 명확하고 일반적인 음식 → 0.6~0.7
      소스/양념에 덮여 재료 불명확   → 0.3~0.4
      용량 추정 불가               → 0.2 이하
    - 추후 식약처 영양성분 DB와 매칭 예정이므로 수치는 표준 단위(g, mg, kcal) 사용
    - 음식이 아닌 이미지면 analysis_status를 failed로, fail_reason에 한글로 사유 작성
    - 의료 진단 및 영양 처방 금지
    - confidence는 음식 인식 확실성 기준으로 유동적으로 설정
      이미지에서 명확히 식별 가능  → 0.85~0.95
      대략적으로 식별 가능         → 0.65~0.80
      소스/양념에 덮여 불분명      → 0.40~0.60
      거의 식별 불가               → 0.40 이하
    """,

    AnalysisType.PRESCRIPTION: """
이 약 봉투 또는 처방전 이미지에서 약물 정보를 추출하세요. 반드시 아래 JSON만 응답하세요 (마크다운 금지):
{
  "medications": [
    {
      "drug_name": "약품명",
      "dosage": "용량 (예: 5mg)",
      "quantity": "수량",
      "confidence": 0.0~1.0,
      "raw_text": "이미지 원문 텍스트"
    }
  ],
  "analysis_status": "success|partial|failed",
  "requires_manual_input": ["인식 불확실 항목"],
  "fail_reason": null
}

규칙:
- 복용법(횟수, 식전/후/간)은 추출하지 말 것 — 사용자가 직접 선택
- 글자가 흐리거나 잘 안 보이면 partial로 반환
- 처방전이 아닌 이미지면 failed로 반환하고 fail_reason 작성
- 의료 진단 및 처방 변경 권고 금지
""",

    AnalysisType.CHECKUP: """
이 건강검진 결과지에서 아래 항목의 수치만 추출하세요. 반드시 아래 JSON만 응답하세요 (마크다운 금지):
{
  "extracted_data": {
    "systolic_bp": null또는숫자,
    "diastolic_bp": null또는숫자,
    "fasting_glucose": null또는숫자,
    "hba1c": null또는숫자,
    "total_cholesterol": null또는숫자,
    "triglyceride": null또는숫자,
    "hdl": null또는숫자,
    "ldl": null또는숫자,
    "height_cm": null또는숫자,
    "weight_kg": null또는숫자,
    "bmi": null또는숫자,
    "waist_cm": null또는숫자
  },
  "confidence_per_field": {},
  "analysis_status": "success|partial|failed",
  "unreadable_fields": [],
  "fail_reason": null
}

규칙:
- 수치가 없거나 읽기 어려우면 null
- 건강검진 결과지가 아닌 이미지면 failed로 반환하고 fail_reason 작성
- 정상/비정상 판정 절대 금지 — 수치만 추출
- 단위 변환 금지 — 원본 단위 그대로
""",
}


# ── Vision 클라이언트 ─────────────────────────────────────────────────────────

class VisionClient:
    """GPT Vision API 호출 클라이언트 (gpt-4o-mini)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def analyze(
        self,
        analysis_type: str,
        image_bytes: bytes,
        media_type: str = "image/jpeg",
    ) -> dict:
        """
        이미지를 분석해 결과 dict를 반환합니다.

        Args:
            analysis_type: 분석 유형 (diet / prescription / checkup)
            image_bytes:   원본 이미지 바이트
            media_type:    MIME 타입 (기본 image/jpeg)

        Returns:
            GPT Vision 응답 JSON dict

        Raises:
            ValueError:        JSON 파싱 실패
            openai.APIError:   OpenAI API 오류
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{media_type};base64,{b64}"
        prompt = PROMPTS[analysis_type]

        logger.info("GPT Vision 분석 시작 | type=%s model=%s", analysis_type, self.model)

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )

        raw_text = response.choices[0].message.content.strip()
        logger.info("GPT Vision 응답 수신 | %s...", raw_text[:80])

        # 마크다운 코드블록 제거 후 JSON 파싱
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("JSON 파싱 실패 | %s\n원문: %s", e, raw_text)
            raise ValueError(f"GPT 응답을 파싱할 수 없습니다: {e}") from e
