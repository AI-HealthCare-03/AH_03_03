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
    PRESCRIPTION = "prescription"
    MEDICATION_BAG = "medication_bag"
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

[음식명]
- 조리법, 시즈닝, 소스, 브랜드가 보이면 반드시 음식명에 포함할 것
  예) 뿌링클 순살치킨, 간장 계란밥, 토마토 소스 파스타, 된장찌개
- 같은 재료라도 조리법이 다르면 반드시 구분
  예) 닭고기 → 치킨(튀김) / 닭가슴살(구이/찜) 구분
- 정확히 모르면 가장 유사한 음식명으로 추정할 것. 절대 "튀김", "음식" 같은 단순 단어로 반환 금지

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
    AnalysisType.MEDICATION_BAG: """
이 약봉투 이미지에서 약품명만 추출하세요. 반드시 아래 JSON만 응답하세요 (마크다운 금지):
{
  "medications": [
    {
      "drug_name": "핵심 약품명만 (브랜드명+제형. 예: '암브로콜시럽', '씨앤유캡슐'. 용량 숫자·단위(mg/g/mL/% 등)·괄호 안 부가설명·제조사명은 제외)",
      "confidence": 0.0~1.0,
      "raw_text": "이미지에서 해당 약품명 주변 원문 텍스트"
    }
  ],
  "analysis_status": "success|partial|failed",
  "fail_reason": null
}

중요 — 글자를 "지어내지" 말 것:
- raw_text에는 이미지에서 실제로 보이는 글자만, 보이는 순서 그대로 옮겨 적을 것. drug_name은 반드시 그 raw_text에서 그대로 가져오거나 다듬은 값이어야 하며, raw_text에 없는 단어를 drug_name에 새로 등장시키지 말 것
- 글자가 흐리거나 가려져 있을 때, "이런 모양이면 보통 이런 약/건강식품이겠지"처럼 익숙한 제품명·성분명으로 대체하거나 완성하지 말 것 (예: 흐릿한 캡슐 약을 보고 실제로 인쇄된 글자 대신 "밀크씨슬추출분말", "마그네슘 하이드록사이드"처럼 일반적인 건강기능식품 이름으로 채워 넣는 것 금지)
- 일부 글자만 확실히 보이면, 보이는 글자까지만 적고 나머지는 추측하지 말 것 (예: "○○실○정"처럼 빈 부분은 채우지 말고 보이는 부분만)
- 한글 자모(받침, 비슷하게 생긴 글자: ㅓ/ㅕ, ㅁ/ㅂ, ㅌ/ㅋ 등)를 혼동하기 쉬우니, 확신이 서지 않는 글자는 가장 비슷해 보이는 하나를 고르되 confidence를 낮게 책정할 것
- 약품명을 확신할 수 없으면 confidence를 낮게(0.5 미만) 매기고 analysis_status를 partial로 반환할 것 — 모르는 것을 그럴듯하게 채워 넣는 것보다 "모른다"고 답하는 편이 낫다

"앞부분 우선" 전략 (글씨 크기와 무관):
- 약봉투에 인쇄된 약품 관련 글자열에서, 앞부분이 약의 핵심 이름(브랜드명+제형)을 나타내고 그 뒤에 함량(예: 50mg)·강도·제조사명 등 부가 정보가 이어지는 경우가 많다. 글씨 크기나 굵기는 신경 쓰지 말고, 이 "앞부분 = 핵심 이름" 구조만 염두에 둘 것
- 뒷부분(부가 정보)이 흐리거나 확신이 없을 때, 전체를 끝까지 추측해서 완성하려 하지 말고 "확실하게 읽히는 앞부분의 핵심 이름까지만" drug_name으로 적을 것 (예: "신플랙스세이프정"으로 추정되지만 뒷글자가 불확실하면, 뒷글자를 지어내 "신플렉스정"처럼 잘못 완성하는 것보다 확실한 앞부분 "신플랙스"까지만 적는 편이 정답에 더 가깝다)
- 단, 앞부분만 적더라도 그 글자들 자체는 실제로 보이는 글자와 정확히 일치해야 한다 — "앞부분만 적어도 된다"는 것이 "앞부분도 대충 비슷하게 지어내도 된다"는 뜻은 아니다

그 외 규칙:
- 약품명만 추출할 것. 용량(mg 등), 수량, 복용법(횟수, 식전/후/간), 복용 시기는 절대 추출하지 말 것
- 봉투에 약품명이 여러 개 적혀 있으면 모두 각각의 항목으로 반환할 것
- 글자가 흐리거나 일부만 보이면 보이는 부분까지만 추출하고 analysis_status는 partial로 반환
- 약봉투가 아닌 이미지(영수증, 처방전, 음식 등)면 failed로 반환하고 fail_reason 작성
- 약품명이 아닌 약국명, 환자명, 날짜, 주의사항 문구 등은 추출하지 말 것
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
            # strict=False: GPT가 문자열 값 안에 이스케이프 없는 줄바꿈 등 제어문자를
            # 그대로 포함해 응답하는 경우가 있어, 이를 파싱 실패로 처리하지 않도록 완화한다.
            parsed: dict[str, Any] = json.loads(cleaned, strict=False)
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
                "medication_count": len(output.get("medications") or []),
                "extracted_field_count": len(output.get("extracted_data") or {}),
            },
        )
    except Exception:
        logger.warning("GPT Vision Langfuse trace 기록 실패", exc_info=True)
