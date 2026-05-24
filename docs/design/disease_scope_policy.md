# 질병 분석 범위 정책

이 문서는 1차 풀서비스/시연 기준으로 어떤 질병군이 공식 분석 결과에 저장되고, 어떤 질병군이 ML 모델 또는 참고 분류로만 사용되는지 정리한다. 의료 진단이나 확진 정책이 아니라 서비스 데이터 구조와 표시 범위 정책이다.

## 1. 공식 분석 결과 저장 대상

`analysis_results`에 공식 저장하는 질병군은 아래 4개다.

| 공식 코드 | 화면 표기 | 저장 위치 |
| --- | --- | --- |
| `DIABETES` | 당뇨 | `analysis_results.analysis_type` |
| `HYPERTENSION` | 고혈압 | `analysis_results.analysis_type` |
| `DYSLIPIDEMIA` | 이상지질혈증 | `analysis_results.analysis_type` |
| `OBESITY` | 비만 | `analysis_results.analysis_type` |

## 2. CatBoost ML 적용 대상

현재 최종 CatBoost artifact가 준비된 질병군은 아래 3개다.

| 모델 코드 | 공식 분석 타입 | 현재 처리 |
| --- | --- | --- |
| `DM` | `DIABETES` | CatBoost |
| `HTN` | `HYPERTENSION` | CatBoost |
| `DL` | `DYSLIPIDEMIA` | CatBoost |

정밀 분석(`PRECISION`)에서는 artifact 로드와 feature mapping이 성공하면 위 3개 질병군에 CatBoost 예측값을 반영한다. artifact가 없거나 추론 실패가 발생하면 API는 중단하지 않고 rule-based fallback으로 내려갈 수 있으며, 실패 원인은 서버 로그로 추적한다.

## 3. OBESITY 처리

`OBESITY`는 공식 `analysis_results` 저장 대상이지만 현재 CatBoost artifact가 없다.

따라서 1차 범위에서는 비만 위험도는 `rule_based` 방식으로 계산한다. 비만 ML 모델을 공식 분석에 추가하려면 별도 artifact, feature schema, threshold, 검증 지표, 테스트를 추가해야 한다.

## 4. ANEM 처리

`ANEM` 또는 `ANEMIA`는 현재 공식 `analysis_results` 저장 대상이 아니다.

현재 사용 위치는 아래 참고 분류에 한정한다.

- X2 health stage classifier의 참고 분류
- 식단 질병군별 점수화의 참고 점수

빈혈을 공식 분석 결과에 포함하려면 아래 변경이 필요하다.

- `AnalysisType` enum에 `ANEMIA` 또는 대응 코드 추가
- DB schema/migration 반영
- DTO와 API 응답 반영
- 분석 서비스 저장 로직 반영
- 프론트 분석 결과/대시보드 UI 반영
- 테스트와 seed 데이터 보강

## 5. 발표/시연 표현 원칙

- `DM / HTN / DL`은 CatBoost 기반 참고용 위험도 분석이라고 설명한다.
- `OBESITY`는 현재 rule-based 참고용 위험도 분석이라고 설명한다.
- `ANEM`은 공식 분석 결과가 아니라 X2/식단 점수 참고 분류라고 설명한다.
- 모든 결과는 의료 진단, 확진, 처방이 아니며 건강관리 참고용 안내다.
