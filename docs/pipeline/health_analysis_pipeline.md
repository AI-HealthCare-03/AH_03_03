# 건강위험 분석 파이프라인

## 목적

사용자가 입력한 건강정보를 기준으로 당뇨 및 고혈압 위험도를 분석하고, 주요 위험요인과 추천 챌린지를 생성한다. Backend는 요청/조회/저장을 담당하고, AI Worker는 모델 입력 생성, 추론, SHAP 또는 feature contribution 계산을 담당한다.

## 입력값

- `user_id`: 분석 요청 사용자
- `health_record_id`: 분석 기준 건강정보
- 건강정보 주요 항목:
  - 키, 몸무게, BMI
  - 수축기/이완기 혈압
  - 공복혈당, HbA1c
  - 당뇨/고혈압 질환 여부
  - 흡연, 음주, 운동 빈도, 수면 시간

## 처리 흐름

1. 사용자가 건강정보를 입력한다.
2. `health_records`에 최신 건강정보를 저장한다.
3. 분석 요청 시 `async_jobs`를 생성한다.
4. `health_record_id` 기준으로 모델 입력 피처를 생성한다.
5. 당뇨 위험도와 고혈압 위험도를 계산한다.
6. SHAP 또는 feature contribution 값을 생성한다.
7. `analysis_results`에 최종 위험도 결과를 저장한다.
8. `analysis_result_factors`에 주요 위험요인을 저장한다.
9. `analysis_snapshots`에 입력/출력/SHAP 원본을 저장한다.
10. `challenge_recommendations`에 추천 챌린지를 저장한다.
11. 필요 시 `notifications`에 웹 내부 알림을 생성한다.
12. Dashboard API는 별도 테이블 없이 관련 테이블을 실시간 조회/집계한다.

## 출력 테이블

- `async_jobs`: 분석 작업 생성 및 상태 관리
- `analysis_results`: 당뇨/고혈압 위험도 최종 결과
- `analysis_result_factors`: 주요 위험요인 및 SHAP 기여도
- `analysis_snapshots`: 분석 입력/출력/SHAP 원본
- `challenge_recommendations`: 분석 결과 기반 추천 챌린지
- `notifications`: 분석 완료 또는 추천 생성에 대한 웹 내부 알림

## AI Worker와 Backend 역할 분리

Backend 담당:

- 사용자 인증 및 권한 확인
- 건강정보 입력/조회/수정 API
- 분석 요청 API
- `async_jobs` 생성
- 분석 결과 조회 API
- Dashboard 조회/집계 API
- 내부 알림 조회 API

AI Worker 담당:

- `async_jobs` 기반 분석 작업 처리
- `health_record_id` 기준 모델 입력 피처 생성
- ML 모델 추론
- SHAP 또는 feature contribution 계산
- 분석 결과, 위험요인, 스냅샷, 추천 챌린지 저장
- 작업 성공/실패 상태 갱신

## SHAP Factor 저장 기준

- 저장 대상은 사용자에게 설명 가능한 상위 위험요인으로 제한한다.
- `factor_key`는 코드에서 사용하는 안정적인 영문 키를 사용한다.
- `factor_name`은 화면 표시용 한글명을 사용한다.
- `factor_value`는 수치, 범주, 문장 값을 모두 담을 수 있도록 varchar로 저장한다.
- `contribution_score`는 SHAP 또는 feature contribution 값을 저장한다.
- `direction`은 위험도를 높이면 `POSITIVE`, 낮추면 `NEGATIVE`, 판단 불가 시 `NEUTRAL`을 사용한다.
- 화면 노출 순서는 `display_order`로 제어한다.

## Challenge Recommendation 생성 기준

- 위험도 높은 분석 결과를 우선 고려한다.
- SHAP factor 중 기여도가 큰 항목을 추천 근거로 사용한다.
- 혈압 관련 위험요인은 혈압/운동/생활습관 챌린지와 연결한다.
- 혈당 관련 위험요인은 혈당/식습관/운동 챌린지와 연결한다.
- 이미 참여 중이거나 최근 완료한 챌린지는 중복 추천하지 않는 것을 원칙으로 한다.
- MVP에서는 정교한 개인화 랭킹보다 규칙 기반 추천을 우선한다.

## Dashboard 집계 기준

Dashboard는 별도 저장 테이블을 만들지 않고 다음 테이블을 실시간 조회/집계한다.

- 최근 건강정보: `health_records`
- 최신 위험도 결과: `analysis_results`
- 주요 위험요인: `analysis_result_factors`
- 챌린지 진행 상태: `user_challenges`
- 추천 챌린지: `challenge_recommendations`
- 읽지 않은 알림 수: `notifications`

## MVP에서 하지 않는 것

- DIET 실서비스 DB 설계
- LLM 답변 실서비스 DB 저장
- FAMILY, QNA, MEDICATION, ADMIN 기능
- 운영 로그 상세 저장
- 외부 SMS/Email/Push/Kakao 알림 발송
- 실시간 스트리밍 분석
- 복잡한 모델 registry
- 독립적인 queue abstraction 설계
- 관리자용 분석 모니터링 화면
