# LLM 구조 및 실험 정리

## 1. 현재 LLM 구조

현재 LLM 영역은 크게 두 축으로 구성한다.

1. 메인 건강 챗봇
2. ML 모델 결과 기반 추천 문구 및 챌린지 안내

코드상으로는 두 번째 축에서 파생된 "결과 기반 챗봇"도 함께 존재한다. 따라서 발표나 문서에서는 크게 2개 LLM 기능으로 설명하되, 구현 상세에서는 결과 기반 챗봇을 ML 결과 기반 기능의 확장으로 정리하는 것이 자연스럽다.

### 1.1 메인 건강 챗봇

메인 건강 챗봇은 사용자의 일반적인 건강 질문에 답하는 기능이다. 사용자의 현재 ML 분석 결과를 직접 전제로 하기보다는, 만성질환 예방과 생활습관 관리 관점에서 안내한다.

관련 파일:

- `ai_worker/llm/health_chatbot.py`
- `ai_worker/llm/rule_engine.py`
- `ai_worker/llm/llm_generator.py`
- `ai_worker/llm/prompt_templates.py`
- `ai_worker/llm/response_router.py`

입력 스키마:

- `MainHealthChatbotInput`
  - `user_message`: 사용자 질문
  - `tone`: 응답 톤, 기본값 `friendly`

출력 스키마:

- `MainHealthChatbotOutput`
  - `answer`: 최종 답변
  - `intent`: 질문 의도
  - `source`: 응답 생성 방식
  - `caution_message`: 의료 주의 문구
  - `tone`: 응답 톤
  - `is_safe`: 안전성 검사 결과
  - `safety_result`: 안전성 검사 상세

현재 intent 분류:

- `hypertension_guidance`: 고혈압/혈압 관련 질문
- `diabetes_guidance`: 당뇨/혈당 관련 질문
- `dyslipidemia_guidance`: 이상지질혈증/콜레스테롤/중성지방 관련 질문
- `obesity_guidance`: 비만/체중/BMI 관련 질문
- `chronic_disease_prevention`: 일반 만성질환 예방 질문
- `medical_consult_required`: 약물, 치료, 처방, 복용/중단 관련 질문

현재 동작 방식:

1. `response_router.py`에서 메인 챗봇 응답을 라우팅한다.
2. 기본적으로 `health_chatbot.py`와 `rule_engine.py`가 키워드 기반으로 intent를 분류하고 답변을 만든다.
3. `use_llm_rewrite=True`이면 룰엔진 답변을 LLM으로 더 자연스럽게 다듬는 흐름을 탄다.
4. `use_llm_fallback=True`이고 룰엔진이 처리하지 못한 질문이면 LLM fallback 응답을 사용할 수 있다.
5. `use_real_llm=True`일 때 실제 OpenAI API를 호출하고, 기본 모델은 `gpt-4o-mini`이다.

프롬프트 방향:

- 의료 진단, 확진, 치료, 처방, 약물 복용/중단 판단을 하지 않는다.
- 고혈압, 당뇨, 이상지질혈증, 비만 관련 질문은 생활습관 관리 관점으로 답한다.
- 약물/치료 질문은 의료진 상담을 권고한다.
- 모든 답변에 진단이 아니며 의료진 상담이 필요하다는 의미를 포함한다.

### 1.2 ML 모델 결과 기반 추천 문구 및 챌린지 안내

이 기능은 ML 모델의 질환별 예측 결과를 바탕으로 위험요인과 추천 챌린지를 구성하고, 사용자에게 보여줄 요약 문구를 생성한다.

관련 파일:

- `ai_worker/llm/risk_mapper.py`
- `ai_worker/llm/recommendation_message.py`
- `ai_worker/llm/schemas.py`
- `ai_worker/llm/prompt_templates.py`
- `ai_worker/llm/notebooks/01_recommendation_message_experiment.ipynb`
- `ai_worker/llm/experiments/run_health_chatbot_poc.py`

입력 흐름:

1. ML 모델이 질환별 예측 결과를 만든다.
2. `DiseasePredictionSet` 형태로 고혈압, 당뇨, 이상지질혈증, 비만 예측값을 전달한다.
3. `risk_mapper.py`가 `pred == 1`인 질환을 위험요인과 챌린지로 매핑한다.
4. `recommendation_message.py`가 사용자에게 보여줄 추천 문구를 생성한다.

현재 질환별 매핑:

| 질환 | 위험요인 이름 | 추천 챌린지 |
| --- | --- | --- |
| 고혈압 | 혈압 | 짠 음식 줄이기 |
| 당뇨 | 혈당 | 단 음료 줄이기 |
| 이상지질혈증 | 지질 수치 | 포화지방 줄이기 |
| 비만 | BMI | 하루 7000보 걷기 |

위험군 생성 방식:

- 위험 질환이 없으면 `risk_group = low_risk`
- 위험 질환이 하나 이상이면 질환명을 조합해 `risk_group`을 만든다.
- 예: `diabetes_obesity_risk`, `hypertension_diabetes_dyslipidemia_obesity_risk`

추천 문구 출력:

- `summary_message`: 입력된 건강정보 기준의 요약 문구
- `challenge_message`: 추천 챌린지 안내 문구
- `caution_message`: 건강관리 참고용 안내 및 의료진 상담 문구
- `tone`: 응답 톤
- `is_safe`: 안전성 검사 결과

현재 구현상 `recommendation_message.py`는 실제 LLM 호출보다는 안정적인 룰 기반 생성에 가깝다. 다만 노트북에서는 실제 LLM 프롬프트를 실험했고, 그 결과를 바탕으로 `RECOMMENDATION_MESSAGE_V2_1_SYSTEM_PROMPT`가 정리되어 있다.

### 1.3 결과 기반 챗봇

결과 기반 챗봇은 ML 분석 결과 화면에서 사용자가 추가 질문을 했을 때 답하는 기능이다. 큰 분류로는 "ML 모델 결과 기반 LLM 기능"에 포함된다.

관련 파일:

- `ai_worker/llm/health_chatbot.py`
- `ai_worker/llm/rule_engine.py`
- `ai_worker/llm/llm_generator.py`
- `ai_worker/llm/grounding.py`
- `ai_worker/llm/response_router.py`

입력 스키마:

- `ResultChatbotInput`
  - `user_message`: 사용자 질문
  - `risk_factors`: ML 결과 기반 위험요인
  - `recommended_challenges`: 추천 챌린지
  - `tone`: 응답 톤

현재 intent 분류:

- `diabetes_result_guidance`: 당뇨/혈당 결과 관련 안내
- `hypertension_result_guidance`: 고혈압/혈압 결과 관련 안내
- `challenge_recommendation`: 챌린지 추천 관련 안내
- `health_metric_explanation`: 수치 해석 관련 안내
- `health_status_summary`: 위험/상태 요약
- `general_health_guidance`: 일반 건강관리 안내
- `medical_consult_required`: 약물, 치료, 처방, 복용/중단 관련 질문

안전장치:

- `safety.py`에서 의료적으로 위험한 표현을 검사한다.
- `grounding.py`에서 입력에 없는 위험요인, 챌린지, 숫자를 답변에 추가하지 않도록 확인한다.
- LLM rewrite가 안전성 또는 grounding 검사를 통과하지 못하면 룰엔진 답변으로 fallback한다.

## 2. 지금까지 진행한 LLM 실험

### 2.1 추천 문구 생성 실험

실험 위치:

- `ai_worker/llm/notebooks/01_recommendation_message_experiment.ipynb`
- `ai_worker/llm/recommendation_message.py`
- `ai_worker/llm/prompt_templates.py`

실험 목적:

- ML 분석 결과와 추천 챌린지를 바탕으로 사용자에게 보여줄 짧은 건강관리 안내 문구를 생성한다.
- 출력은 `summary_message`, `challenge_message`, `caution_message`, `tone` 구조로 고정한다.
- 진단/확진/치료/처방 표현을 피하고, 건강관리 참고용 안내로 제한한다.

실험 흐름:

1. 위험요인과 추천 챌린지가 포함된 샘플 입력을 구성했다.
2. LLM 프롬프트 v1을 만들어 추천 문구를 생성했다.
3. v1 결과에서 주요 위험요인과 챌린지 이유 반영이 약한 점을 확인했다.
4. v2에서 `main_risk_factors`와 `recommended_challenges`의 title/reason을 더 구체적으로 반영하도록 개선했다.
5. v2.1에서 의료 표현을 더 완화하고, "입력된 건강정보 기준으로", "도움이 될 수 있습니다", "의료진 상담 필요" 같은 안전한 문장을 강화했다.

실험 케이스 예시:

- 당뇨 high / 고혈압 medium / 추천 챌린지 있음
- 위험 신호 낮음 / 기본 챌린지 있음
- 당뇨 medium / 고혈압 medium / 챌린지 없음
- 혈당/혈압 관련 위험요인과 챌린지가 함께 있는 케이스
- 혈압 관련 단일 위험요인 케이스

정리된 프롬프트:

- `RECOMMENDATION_MESSAGE_V2_1_SYSTEM_PROMPT`

현재 판단:

- 추천 문구 생성은 실험상 LLM을 사용할 수 있도록 설계했다.
- 실제 코드에서는 안정성과 재현성을 위해 룰 기반 생성 함수가 먼저 구현되어 있다.
- 추후 실제 LLM 적용 시에는 노트북의 v2.1 프롬프트를 모듈화하고, JSON schema 검증과 safety check를 함께 붙이는 방식이 적합하다.

### 2.2 메인 건강 챗봇 실험

실험 위치:

- `ai_worker/llm/experiments/run_health_chatbot_poc.py`
- `ai_worker/llm/experiments/dummy_health_cases.py`
- `ai_worker/llm/health_chatbot.py`
- `ai_worker/llm/llm_generator.py`
- `ai_worker/llm/response_router.py`

실험 목적:

- 사용자의 일반 건강 질문을 만성질환 생활습관 관리 관점에서 답변한다.
- 고혈압, 당뇨, 이상지질혈증, 비만 질문을 intent별로 분류한다.
- 약물, 치료, 처방, 복용 중단 관련 질문은 직접 답하지 않고 의료진 상담으로 유도한다.

실험 케이스 예시:

- "당뇨가 있으면 뭘 조심해야 하나요?"
- "혈압약 끊어도 되나요?"

실험 결과 구조:

- 기본 룰엔진 응답
- LLM fallback 응답
- LLM rewrite 응답
- safety check 결과

현재 판단:

- 메인 건강 챗봇은 룰엔진만으로도 MVP 수준의 핵심 질환 질문을 처리할 수 있다.
- LLM은 복잡하거나 자연스러운 표현이 필요한 질문에서 fallback 또는 rewrite 용도로 붙이는 구조가 적합하다.
- 약물/치료 관련 질문은 LLM을 사용하더라도 의료진 상담 안내를 유지해야 한다.

### 2.3 ML 결과 기반 챗봇 실험

실험 위치:

- `ai_worker/llm/experiments/run_health_chatbot_poc.py`
- `ai_worker/llm/experiments/run_rule_vs_llm_comparison.py`
- `ai_worker/llm/experiments/dummy_health_cases.py`
- `ai_worker/llm/grounding.py`
- `ai_worker/llm/llm_generator.py`

실험 목적:

- 사용자가 ML 결과 화면에서 "왜 이런 추천을 받았는지", "무엇을 해야 하는지"를 질문했을 때 답변한다.
- 답변은 입력으로 받은 위험요인과 추천 챌린지 범위 안에서만 생성한다.
- 검사 수치나 예측 확률을 직접 단정적으로 말하지 않도록 제한한다.

실험 케이스 예시:

- "당뇨 위험이 있다는데 뭘 해야 하나요?"
- "혈압이 높다는데 왜 걷기를 추천했나요?"

검증 포인트:

- 입력에 없는 질환, 위험요인, 챌린지를 추가하지 않는지 확인
- 약물/치료/처방 질문에서 직접 판단하지 않는지 확인
- 숫자 수치를 직접 노출하지 않는지 확인
- 안전 문구가 유지되는지 확인

현재 판단:

- 결과 기반 챗봇은 추천 문구 생성 기능의 확장으로 볼 수 있다.
- 사용자가 결과를 이해하고 챌린지를 실천하도록 돕는 역할이다.
- LLM rewrite를 사용할 경우 grounding 검사를 반드시 통과해야 한다.

### 2.4 Rule Engine vs LLM 비교 실험

실험 위치:

- `ai_worker/llm/experiments/run_rule_vs_llm_comparison.py`

실험 목적:

- 동일한 입력에 대해 룰엔진, LLM 직접 응답, LLM rewrite, 라우터 결과를 비교한다.
- 실제 LLM 호출 여부를 `--use-real-llm` 옵션으로 제어한다.
- 여러 케이스를 한 번에 호출할 때는 `--confirm-all` 옵션을 요구해 의도치 않은 API 호출을 방지한다.

비교 대상:

- `rule_engine`
- `llm` 또는 `llm_stub`
- `llm_rewrite` 또는 `llm_rewrite_stub`
- `route_*_chatbot_response`

현재 판단:

- MVP에서는 룰엔진을 기본값으로 두고, LLM은 fallback/rewrite로 붙이는 방식이 안전하다.
- 실제 LLM을 사용할 때는 safety check와 grounding check 결과를 함께 기록해야 한다.
- LLM rewrite 실패 시 룰엔진 답변으로 되돌리는 fallback 구조가 이미 잡혀 있다.

### 2.5 전체 정리

현재까지의 LLM 실험은 "LLM이 건강 위험을 새로 판단하는 구조"가 아니라, "ML 모델과 룰 기반 판단 결과를 사용자 친화적으로 설명하는 구조"로 설계되었다.

핵심 원칙:

- 질환 위험 판단은 ML 모델과 사전 정의된 mapping이 담당한다.
- LLM은 진단자가 아니라 설명자 역할을 한다.
- 입력에 없는 질환, 수치, 위험요인, 챌린지를 추가하지 않는다.
- 의료적 결정이 필요한 질문은 의료진 상담으로 연결한다.
- MVP에서는 룰 기반 응답을 기본으로 두고, LLM은 fallback 또는 rewrite로 점진 적용한다.

현재 구조를 두 문장으로 요약하면 다음과 같다.

- 메인 화면에서는 건강 Q&A 챗봇이 사용자의 일반 질문에 생활습관 관리 중심으로 답한다.
- ML 결과 화면에서는 예측 결과를 위험요인과 챌린지로 변환한 뒤, 추천 문구와 결과 기반 Q&A로 사용자가 이해하고 실천할 수 있게 돕는다.
