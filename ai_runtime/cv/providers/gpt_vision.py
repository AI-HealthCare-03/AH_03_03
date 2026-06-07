"""
    ai_runtime/cv/providers/gpt_vision.py

GPT Vision API 호출 클라이언트.
프롬프트 관리 및 이미지 분석 요청을 담당합니다.
"""

import base64
import io
import json
import logging
from typing import Any

from openai import AsyncOpenAI
from PIL import Image


def _upscale_image(image_bytes: bytes, scale: float = 2.0) -> bytes:
    """이미지 해상도를 업스케일합니다."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


logger = logging.getLogger(__name__)


class AnalysisType:
    DIET = "diet"
    MEDICATION = "medication"
    PRESCRIPTION = "prescription"
    CHECKUP = "checkup"


PROMPTS: dict[str, str] = {
    AnalysisType.DIET: """
이 식단 이미지를 분석하세요. 반드시 아래 JSON만 응답하세요 (마크다운 금지):
{
  "foods": [
    {
      "name": "음식명 (조리법·시즈닝·브랜드 포함, 예: 뿌링클 순살치킨, 된장찌개, 계란프라이)",
      "nutrient_category": "탄수화물|단백질|지방|식이섬유|비타민|미네랄|항산화|건강식",
      "cooking_method": "조리법 (예: 튀김, 구이, 찜, 볶음, 생것)",
      "estimated_amount": "1인분 기준 추정 용량 (예: 200g, 1공기, 10개(300g))",
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

[음식명 - 가장 중요]
- 반드시 구체적인 한국 음식명을 사용할 것. 상위 개념 금지
  틀린 예) "조개구이" → 맞는 예) "가리비구이" 또는 "홍합구이" (종류를 특정)
  틀린 예) "떡" → 맞는 예) "가래떡" 또는 "찹쌀떡" 또는 "시루떡" (종류를 특정)
  틀린 예) "과일" → 맞는 예) "감" 또는 "사과" 또는 "귤" (종류를 특정)
  틀린 예) "국" → 맞는 예) "갈비탕" 또는 "된장찌개" (요리명을 특정)
- 조리법, 시즈닝, 소스, 브랜드가 보이면 반드시 음식명에 포함할 것
  예) 뿌링클 순살치킨, 간장 계란밥, 토마토 소스 파스타, 된장찌개
- 같은 재료라도 조리법이 다르면 반드시 구분
  예) 닭고기 → 치킨(튀김) / 닭가슴살(구이/찜) 구분
- 시각적 특징(색, 모양, 질감, 용기)으로 최대한 구체적으로 특정할 것
  예) 주황빛 국물+뼈 → 갈비탕, 흰 국물+두부 → 순두부찌개
- 정확히 모르면 가장 유사한 음식명으로 추정할 것. 절대 "튀김", "음식", "요리" 같은 단순 단어로 반환 금지

[1인분 기준 용량 추정]
- estimated_amount는 항상 1인분 기준으로 추정할 것
- 이미지에서 용량 파악이 어려워도 일반적인 1인분 기준으로 추정하여 작성
  예) 치킨 → "10조각(300g)", 밥 → "1공기(210g)", 찌개 → "1인분(400g)"
- amount_requires_input은 용량 추정이 매우 불확실한 경우에만 true

[영양성분 추정]
- nutrition은 estimated_amount 기준 1인분 영양성분을 반드시 추정할 것
- 정확한 수치를 모르면 일반적인 해당 음식의 평균값으로 추정
- 추정 불가한 항목만 null 허용. 칼로리, 단백질, 탄수화물, 지방은 반드시 추정값 제공
- 4대 만성질환 관련 영양소 반드시 포함
  고혈압       → 나트륨, 식이섬유
  당뇨         → 당류, 탄수화물, 식이섬유
  이상지질혈증  → 포화지방, 식이섬유
  비만         → 칼로리, 지방
- 영양성분_신뢰도 기준:
  음식명이 명확하고 일반적인 음식 → 0.6~0.7
  소스/양념에 덮여 재료 불명확   → 0.3~0.5
  용량 추정 불확실               → 0.2~0.4

[nutrient_category]
  탄수화물 → 밥, 빵, 면류, 떡
  단백질   → 닭가슴살, 두부, 생선, 계란
  지방     → 삼겹살, 튀김류, 버터
  식이섬유 → 고구마, 양배추, 당근
  비타민   → 브로콜리, 파프리카, 시금치
  미네랄   → 시금치, 견과류, 콩류
  항산화   → 토마토, 블루베리, 당근
  건강식   → 아보카도, 연어, 올리브유

[analysis_status]
- 음식이 하나라도 인식되면 success 또는 low_confidence로 반환
- failed는 음식이 전혀 없는 이미지(영수증, 풍경 등)일 때만 사용
- 음식이 불분명해도 low_confidence로 반환하고 최대한 추정할 것

- 의료 진단 및 영양 처방 금지
""",
    AnalysisType.MEDICATION: """
이 약봉투 이미지에서 약물 정보를 추출하세요. 반드시 아래 JSON만 응답하세요 (마크다운 금지):
{
  "medications": [
    {
      "drug_name": "약품명",
      "role": "복약안내 (약품 역할 및 기능)",
      "dosage": "투약량 (예: 1정, 5ml)",
      "frequency": "횟수 (예: 하루 3회)",
      "days": "일수 (예: 3일치)",
      "timing": null,
      "confidence": 0.0~1.0,
      "raw_text": "이미지 원문 텍스트"
    }
  ],
  "analysis_status": "success|partial|failed",
  "requires_manual_input": ["인식 불확실 항목"],
  "fail_reason": null
}

규칙:
- timing은 반드시 null로 반환 (사용자 수기 입력)
- 모든 항목은 사용자가 수정 가능한 참고값으로 제공
- 이미지에 있는 모든 약품을 추출할 것 (1개만 추출하는 것은 오류)
- 약품 목록 추출 우선순위: 1) 하단/왼쪽 표(약품명 투약량 횟수 일수) 2) 복약안내 목록
- 표가 있으면 각 행을 하나의 약품으로 추출하고 해당 행의 수치만 사용할 것
- 약품명은 이미지에 보이는 제품명을 글자 그대로 읽을 것. 절대 추측하거나 수정하지 말 것
- 인식 불확실 항목은 null로 반환하고 requires_manual_input에 추가할 것
- role은 이미지에서 읽히는 약품 설명 텍스트로 작성할 것
- 약봉투가 아닌 이미지면 failed로 반환하고 fail_reason 작성
- 의료 진단 및 처방 변경 권고 금지
""",
    AnalysisType.PRESCRIPTION: """
이 처방전 이미지에서 처방 의약품 정보를 추출하세요. 반드시 아래 JSON만 응답하세요 (마크다운 금지):
{
  "medications": [
    {
      "drug_name": "처방 의약품 명칭",
      "single_dose": "1회 투약량 (예: 1정, 0.5정)",
      "daily_frequency": "1일 투여 횟수 (예: 1일 3회)",
      "total_days": "총 투약일수 (예: 3일)",
      "usage": "용법 (예: 식후 30분, 취침 전)",
      "confidence": 0.0~1.0,
      "raw_text": "이미지 원문 텍스트"
    }
  ],
  "analysis_status": "success|partial|failed",
  "requires_manual_input": ["인식 불확실 항목"],
  "fail_reason": null
}

규칙:
- 모든 항목은 사용자가 수정 가능한 참고값으로 제공
- 약품명은 이미지에 보이는 글자를 철자 하나하나 그대로 옮겨 적을 것. 절대 추측하지 말 것
- 1회 투약량, 1일 횟수, 총 투약일수는 표에서 해당 약품 행의 수치만 추출할 것
- 인식 불확실 항목은 null로 반환하고 requires_manual_input에 추가할 것
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

        # 처방전/약봉투는 이미지 업스케일 적용
        if analysis_type in (AnalysisType.PRESCRIPTION, AnalysisType.MEDICATION):
            image_bytes = _upscale_image(image_bytes)

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            temperature=0,
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

        # 토큰 사용량 추출 (Langfuse 비용 추적용)
        usage = response.usage
        token_usage = {
            "input_tokens":  usage.prompt_tokens     if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "total_tokens":  usage.total_tokens      if usage else None,
        }

        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            parsed: dict[str, Any] = json.loads(cleaned)
            _record_vision_trace(
                analysis_type=analysis_type,
                model=self.model,
                media_type=media_type,
                output=parsed,
                success=True,
                token_usage=token_usage,
            )
            return parsed
        except json.JSONDecodeError as e:
            _record_vision_trace(
                analysis_type=analysis_type,
                model=self.model,
                media_type=media_type,
                output={"raw_text_preview": cleaned[:120]},
                success=False,
                error_type=type(e).__name__,
            )
            logger.error("JSON 파싱 실패 | %s\n원문: %s", e, raw_text)
            raise ValueError(f"GPT 응답을 파싱할 수 없습니다: {e}") from e


def _record_vision_trace(
    *,
    analysis_type: str,
    model: str,
    media_type: str,
    output: dict[str, Any],
    success: bool,
    error_type: str | None = None,
    token_usage: dict[str, Any] | None = None,
) -> None:
    """Langfuse 설정이 있을 때만 Vision 호출을 generation 타입으로 기록한다.
    generation 타입으로 기록해야 모델 단가 설정에 따라 비용이 자동 계산된다.
    """
    try:
        from ai_runtime.llm.llm_client import record_langfuse_generation

        record_langfuse_generation(
            name=f"{analysis_type}.gpt_vision",
            model=model,
            output_payload=output,
            input_tokens=token_usage.get("input_tokens") if token_usage else None,
            output_tokens=token_usage.get("output_tokens") if token_usage else None,
            metadata={
                "provider": "gpt_vision",
                "analysis_type": analysis_type,
                "media_type": media_type,
                "success": success,
                "error_type": error_type,
                "analysis_status": output.get("analysis_status"),
                "food_count": len(output.get("foods") or []),
                "medication_count": len(output.get("medications") or []),
                "extracted_field_count": len(output.get("extracted_data") or {}),
            },
        )
    except Exception:
        logger.warning("GPT Vision Langfuse trace 기록 실패", exc_info=True)
