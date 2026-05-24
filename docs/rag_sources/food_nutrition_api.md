# 식품영양성분 API 후보

- status: candidate_unreviewed
- source_org: 식품의약품안전처, 공공데이터포털, 농촌진흥청 농식품올바로
- source_url: https://www.data.go.kr/data/15127578/openapi.do
- runtime_use: provider_metadata_candidate

## 사용 목적

- 식품명과 영양성분을 표준화하는 외부 provider 후보로 검토한다.
- 현재 keyword RAG와 식단 점수 계산은 외부 API를 호출하지 않는다.

## 요약

- 식품영양성분 API는 음식명, 1회 제공량, 열량, 탄수화물, 당류, 단백질, 지방, 포화지방, 나트륨 등 영양성분 정규화 후보로 검토한다.
- 런타임 식단 점수 계산은 현재 `food_disease_scores.csv`와 JSON rule table을 우선 사용한다.
- 외부 API는 비용, 호출 제한, 응답 품질, 음식명 매칭 정확도, 라이선스 검토 후 별도 provider로 연결한다.
- API 응답은 곧바로 사용자에게 노출하지 않고 내부 표준 스키마로 변환한 뒤 DiseaseFoodScorer 입력으로 사용한다.
- 원천 데이터가 없거나 매칭 신뢰도가 낮으면 사용자 확인 또는 수동 입력으로 유도한다.
- 공공데이터포털의 식품의약품안전처 API와 농촌진흥청 농식품올바로 API는 후보로만 관리한다.

## 표준 필드 후보

- food_name
- serving_size
- calories_kcal
- carbohydrate_g
- sugar_g
- protein_g
- fat_g
- saturated_fat_g
- sodium_mg
- cholesterol_mg
- fiber_g

## keyword 후보

식품영양성분, 공공데이터포털, 식품의약품안전처, 농식품올바로, 음식명, 영양성분, 열량, 나트륨, 당류, API provider, 데이터 정규화
