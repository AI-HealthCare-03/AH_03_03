# 식품영양성분 API 후보

- status: candidate_unreviewed
- source_org: 공공데이터포털
- source_url: https://www.data.go.kr
- runtime_use: provider_metadata_candidate

## 요약

- 식품영양성분 API는 음식명, 1회 제공량, 열량, 탄수화물, 당류, 단백질, 지방, 나트륨 등 영양성분을 정규화하는 provider 후보로 검토한다.
- 런타임 식단 점수 계산은 현재 `food_disease_scores.csv`와 JSON rule table을 우선 사용한다.
- 외부 API는 비용, 호출 제한, 응답 품질, 음식명 매칭 정확도, 라이선스 검토 후 별도 provider로 연결한다.
- API 응답은 곧바로 사용자에게 노출하지 않고 내부 표준 스키마로 변환한 뒤 DiseaseFoodScorer 입력으로 사용한다.
- 원천 데이터가 없거나 매칭이 낮으면 사용자 확인 또는 수동 입력으로 유도한다.

## keyword 후보

식품영양성분, 공공데이터포털, 음식명, 영양성분, 열량, 나트륨, 당류, API provider, 데이터 정규화
