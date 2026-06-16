"""
    ai_runtime/cv/providers/gpt_vision.py

GPT Vision API 호출 클라이언트.
프롬프트 관리 및 이미지 분석 요청을 담당합니다.
"""

import base64
import json
import logging
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class AnalysisType:
    DIET = "diet"
    CHECKUP = "checkup"


PROMPTS: dict[str, str] = {
    AnalysisType.DIET: """
이 식단 이미지에서 보이는 완성 음식명 후보만 추출하세요.
반드시 아래 JSON만 응답하세요 (마크다운 금지):
{
  "foods": [
    {
      "name": "음식명",
      "confidence": 0.0
    }
  ],
  "analysis_status": "success|low_confidence|failed",
  "fail_reason": null
}

규칙:
- 한국 서비스에서 검색 가능한 구체적인 한국어 음식명을 우선 사용하세요.
- 식단 사진에서는 완성 음식, 반찬, 음료, 명확한 사이드 메뉴 중심으로 반환하세요.
- 단독 포장식품이나 단독 재료 사진이 아니라면 재료, 소스, 양념, 토핑을 foods에 넣지 마세요.
- "계란", "고기", "소고기", "돼지고기", "닭고기", "고추장", "된장", "간장", "소스", "양념", "야채", "채소"처럼 단독 일반 재료명만 반환하지 마세요.
- 정확한 음식명을 모르겠으면 세부 재료가 아니라 완성 음식 단위의 넓은 후보를 낮은 confidence로 반환하세요.
  예: "비빔밥", "김밥", "찌개류", "볶음밥", "국물 면요리", "중식 면요리", "채소 샐러드", "튀김류"
- 낮은 확신은 빈 배열이 아니라 낮은 confidence 값으로 표현하세요.
- foods=[]는 음식, 음료, 식단으로 볼 수 있는 메뉴가 전혀 보이지 않는 경우에만 사용하세요.
- 이미지에 보이지 않는 음식은 만들지 마세요.
- 칼로리, 탄수화물, 단백질, 지방, 나트륨, 영양성분, 섭취량, 용량은 추정하거나 반환하지 마세요.
- 의료 진단 및 영양 처방 금지
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

    def __init__(self, api_key: str, model: str = "gpt-4o"):
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
            parsed: dict[str, Any] = json.loads(cleaned)
            _record_vision_trace(
                analysis_type=analysis_type,
                model=self.model,
                media_type=media_type,
                output=parsed,
                success=True,
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
) -> None:
    """Langfuse 설정이 있을 때만 Vision 호출 metadata를 남긴다."""
    try:
        from ai_runtime.llm.llm_client import record_langfuse_event

        record_langfuse_event(
            name=f"{analysis_type}.gpt_vision",
            output_payload=output,
            metadata={
                "provider": "gpt_vision",
                "analysis_type": analysis_type,
                "model": model,
                "media_type": media_type,
                "success": success,
                "error_type": error_type,
                "analysis_status": output.get("analysis_status"),
                "food_count": len(output.get("foods") or []),
                "extracted_field_count": len(output.get("extracted_data") or {}),
            },
        )
    except Exception:
        logger.warning("GPT Vision Langfuse trace 기록 실패", exc_info=True)
