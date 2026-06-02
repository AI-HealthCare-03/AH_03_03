# Schema Normalization Plan

이 문서는 현재 데모 운영 범위에서 삭제하거나 마이그레이션하지 않고, 운영 전 정리 후보만 기록합니다.

## 개인정보 동의 truth source

- `user_consents`를 개인정보/민감정보/마케팅 동의 이력의 truth source로 봅니다.
- 회원가입 흐름은 `UserConsent`를 생성하며, 탈퇴 시 해당 사용자 동의 이력을 함께 정리합니다.
- `user_settings.sensitive_data_agreed`, `user_settings.marketing_agreed`는 설정 모델/DTO에 남아 있지만 현재 동의 이력의 기준으로 사용하지 않습니다.
- 운영 전에는 `user_settings`의 동의성 필드를 읽는 화면/API가 있는지 재확인하고, 필요하면 `user_consents` 조회 또는 별도 알림 설정으로 역할을 분리합니다.

## user_settings 중복 동의 필드 사용처

- `app/models/settings.py`: `marketing_agreed`, `sensitive_data_agreed` 필드가 있습니다.
- `app/dtos/settings.py`: 설정 생성/수정/응답 DTO에 동일 필드가 있습니다.
- `frontend/src/pages/SignupPage.tsx`, `frontend/src/api/auth.ts`: 회원가입 요청 payload에는 민감정보/마케팅 동의 값이 포함됩니다.
- `app/services/auth.py`, `app/repositories/user_repository.py`: 회원가입 시 `UserConsent`를 생성합니다.

정리 방향은 `user_consents`를 동의 이력 기준으로 유지하고, `user_settings`의 동의성 필드는 운영 전 제거 또는 알림/마케팅 수신 설정으로 의미를 재정의하는 것입니다.

## diet_photo_results.detected_foods 역할

- `diet_photo_results.detected_foods`는 식단 이미지 분석 provider가 반환한 원본 후보 스냅샷으로 취급합니다.
- `diet_records.detected_foods`는 화면 표시와 질환별 식단 점수 계산에 사용하는 record-level 결과로 취급합니다.
- 현재 프론트 결과 화면은 `diet_records.detected_foods`를 우선 사용하고, 없으면 첫 번째 `diet_photo_results.detected_foods`로 fallback합니다.

운영 전 정리 방향은 `diet_photo_results.detected_foods`를 raw/provider 후보로 명확히 명명하거나 문서화하고, 사용자가 확정한 최종 식단 결과는 `diet_records.detected_foods` 기준으로 통일하는 것입니다.
