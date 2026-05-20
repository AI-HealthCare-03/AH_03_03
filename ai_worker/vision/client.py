"""
ai_worker/vision/client.py

GPT Vision API 호출 클라이언트.
프롬프트 관리 및 이미지 분석 요청을 담당합니다.
"""

import base64
import json
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class AnalysisType:
    DIET = "diet"
    PRESCRIPTION = "prescription"
    CHECKUP = "checkup"


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
    - estimated_amount는 이미지에서 용량 추정이 가능하면 작성, 불가능하면 null
    - amount_requires_input은 용량 추정 불가능하면 true
    - nutrition은 estimated_amount 기준으로 추정, 추정 불가 항목은 null
    - 4대 만성질환 관련 영양소 반드시 포함
      고혈압       → 나트륨, 식이섬유
      당뇨         → 당류, 탄수화물, 식이섬유
      이상지질혈증  → 포화지방, 식이섬유
      비만         → 칼로리, 지방
    - 음식이 아닌 이미지면 analysis_status를 failed로, fail_reason에 한글로 사유 작성
    - 의료 진단 및 영양 처방 금지
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
- 복용법(횟수, 식전/후/간)은 추출하지 말 것
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
    "hb": null또는숫자,
    "total_cholesterol": null또는숫자또는"비해당",
    "triglyceride": null또는숫자또는"비해당",
    "hdl": null또는숫자또는"비해당",
    "ldl": null또는숫자또는"비해당",
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
- 검진 결과지에 "비해당", "해당없음", "-", "N/A" 등으로 표시된 항목은 반드시 "비해당" 문자열로 반환할 것
- 인식 오류 시 null로 반환. 참고치 수치와 혼동 금지
- 모든 수치는 반드시 항목명을 먼저 찾고, 그 항목명과 같은 행 또는 바로 옆에 위치한 수치만 추출할 것

항목명 매핑 기준:
  · systolic_bp    → "수축기", "SBP" 포함 행의 첫 번째 수치
  · diastolic_bp   → "이완기", "DBP" 포함 행의 수치, 또는 혈압 수치에서 "/" 뒤 수치
  · fasting_glucose → "공복혈당", "혈당", "GLU" 포함 행의 수치
                      예: "공복혈당(mg/dL) | 83 | 100.0미만" → 83 (100.0미만은 참고치, 절대 추출 금지)
  · hb             → "혈색소", "Hb", "헤모글로빈" 포함 행의 수치 (단위 g/dL)
                      주의: "당화혈색소(HbA1c)"는 다른 항목이므로 절대 혼동 금지
                      예: "혈색소(g/dL) | 16.1 | 13.0~16.5" → 16.1 (13.0~16.5는 참고치, 추출 금지)
  · total_cholesterol → "총콜레스테롤", "T-CHO" 포함 행의 수치
                        예: "총콜레스테롤(mg/dL) | 비해당 | 200.0미만" → "비해당"
  · hdl            → "고밀도 콜레스테롤", "HDL" 포함 행의 수치
                      예: "고밀도 콜레스테롤(mg/dL) | 49 | 60.0이상" → 49
  · triglyceride   → "중성지방", "TG" 포함 행의 수치
                      예: "중성지방(mg/dL) | 105 | 150.0미만" → 105
  · ldl            → "저밀도 콜레스테롤", "LDL" 포함 행의 수치
                      예: "저밀도 콜레스테롤(mg/dL) | 67 | 130.0미만" → 67
  · height_cm      → "키", "신장", "Height" 포함 행의 수치
  · weight_kg      → "몸무게", "체중", "Weight" 포함 행의 수치
  · bmi            → "체질량지수", "BMI" 포함 행에서 체크박스(□■) 및 범위 표현
                      (18.5미만, 18.5~24.9, 25~29.9, 30이상) 옆 숫자는 모두 무시하고,
                      독립적으로 기재된 실측 수치만 추출. 없으면 null
  · waist_cm       → "허리둘레", "허리" 포함 행의 수치

- 참고치(예: 200.0미만, 60.0이상, 150.0미만, 18.5~24.9, 30이상 등) 숫자는 절대 추출 금지
- 표 레이아웃이나 항목 순서가 달라도 항목명 기준으로 매칭할 것
""",
}


class VisionClient:
    """GPT Vision API 호출 클라이언트."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def analyze(
        self,
        analysis_type: str,
        image_bytes: bytes,
        media_type: str = "image/jpeg",
    ) -> dict:
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

        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("JSON 파싱 실패 | %s\n원문: %s", e, raw_text)
            raise ValueError(f"GPT 응답을 파싱할 수 없습니다: {e}") from e
