# Full Service Scope

이 문서는 `feature/kdu-web` 브랜치 기준으로 MVP 시연 단계를 넘어 풀서비스 1차 구현 범위를 고정하기 위한 기준 문서입니다.
현재 제외/보류 항목은 소셜 로그인, 웨어러블 연동 2개이며, 휴대폰 SMS 인증은 Twilio Verify 기반 구현 대상으로 유지합니다.
요구사항 정의서는 사용자 관점 기능 중심으로 관리하고, NFR/아키텍처/재시도/공통 컴포넌트 같은 구현 세부사항은 [Requirements Refactor Notes](requirements_refactor_notes.md)와 후속 설계 문서로 분리합니다.

## 1. 풀서비스 1차 제외/보류 항목

아래 2개 항목은 풀서비스 1차 범위에서 제외하거나 보류합니다.

- 소셜 로그인
- 웨어러블 연동

## 2. 계속 포함되는 항목

아래 항목은 풀서비스 1차 또는 후속 구현 로드맵에 포함합니다.

- FastAPI JWT 인증
- 이메일 인증
- Twilio Verify 기반 휴대폰 SMS 인증
- 아이디/이메일/휴대폰 중복확인
- 로그인 실패 제한
- 비밀번호 변경/재인증
- 회원탈퇴
- `role` / `is_admin` 단일화
- 건강정보 x1/x2
- 회원가입 기본정보와 건강정보 입력 화면 분리
- 건강검진표 OCR
- 복약/처방전 OCR
- 실제 ML inference
- `model_versions`
- `ai_inference_logs`
- 식단 분석
- 챌린지
- 복약/영양제
- 리마인드 알림
- `notification_logs`
- `reminder_schedules`
- 가족 관리
- 관리자/모니터링
- `request_id` 구조화 로그
- `system_error_logs`
- `sensitive_access_logs`
- deep health check
- 성능 메트릭
- 장애 복구/fallback
- 의료 표현 제한 정책
- 개인정보 처리방침/민감정보 동의
- Langfuse 연동

## 3. NFR 재분류

### 1차 범위 제외/보류 NFR

아래 NFR은 1차 범위에서 제외하거나 보류합니다.

- 소셜 로그인 관련 NFR
- 웨어러블 연동 관련 NFR

### 구현 대상으로 유지하는 NFR

위 2개 영역을 제외한 NFR은 풀서비스 구현 대상으로 유지합니다. 특히 아래 항목은 헬스케어 서비스 운영 기준에서 필수로 관리합니다.

- 인증/인가 보안
- 회원가입/건강정보 입력 책임 분리
- 개인정보 및 민감정보 보호
- 민감정보 접근 감사 로그
- 장애 복구 및 fallback
- 성능 측정과 응답 시간 관리
- 운영 로그와 에러 추적
- 관리자 권한 분리
- 의료 진단으로 오인되지 않도록 하는 표현 제한
- 모델 추론 이력과 모델 버전 추적
- 알림 발송 이력과 리마인드 스케줄 관리

## 4. P0 로드맵

P0는 풀서비스 전환 전에 권한, 보안, 감사, 기본 운영 안정성을 먼저 닫는 단계입니다.

- `role` / `is_admin` 단일화
- 로그인 실패 제한
- 이메일 인증 연결
- 회원가입 입력 범위 축소: 기본정보와 건강정보 분리
- 건강정보 x1/x2 validation 기준 정의
- 비밀번호 변경 시 재인증
- `request_id` 기반 구조화 로그
- `system_error_logs`
- `sensitive_access_logs`
- deep health check
- 관리자 권한 dependency 표준화
- 개인정보/민감정보 정책 문서
- 의료 표현 제한 정책
- 분석 실패 메시지 정책

## 5. P1 로드맵

P1은 실제 AI/OCR/알림을 운영 가능한 비동기 처리 구조로 확장하는 단계입니다.

- `async_jobs`
- `async_job_logs`
- Redis queue/worker
- `model_versions`
- `ai_inference_logs`
- 실제 ML inference
- 실제 OCR 파일 업로드/처리
- 식단 이미지 기반 간편 분석과 사용자 보정
- `notification_logs`
- `reminder_schedules`
- 성능 메트릭 p50/p95/p99
- timeout/fallback

## 6. P2 로드맵

P2는 풀서비스 운영 편의성과 확장 기능을 완성하는 단계입니다.

- 가족 관리 풀세트
- 관리자 콘솔
- 관리자 모니터링
- FAQ/문의 관리자 페이지
- 알림 템플릿
- LLM 비용 추적
- Langfuse trace 연동
- 접근성/반응형 고도화
- 영양성분 DB 매칭과 정밀 식단 점수화

## 7. 구현 기준

- Firebase Auth와 소셜 로그인은 1차 풀서비스 범위에서 제외/보류합니다.
- 현재 인증은 FastAPI JWT와 이메일 인증을 기준으로 유지합니다.
- `users` 테이블은 자체 JWT/이메일 인증 기준으로 정리하며, Firebase Auth 전용 컬럼은 사용하지 않습니다.
- Firebase 또는 소셜 로그인을 재도입할 경우 OAuth/Social Login 설계를 별도 작업으로 다시 진행합니다.
- 소셜 로그인 재도입 시에는 `users` 테이블에 provider 식별자를 직접 넣기보다 `user_oauth_accounts(id, user_id, provider, provider_user_id, email, created_at, updated_at)` 같은 별도 연결 테이블을 권장합니다.
- 회원가입은 아이디, 이메일, 이메일 인증, 비밀번호, 이름, 성별, 생년월일, 휴대폰 번호, 주소, 동의 항목 같은 기본정보 중심으로 유지합니다.
- 주소는 초기에는 서비스 지역 확인용 필수 기본정보로 받되, 상세주소 수집 범위와 보관 정책은 운영 전 개인정보 최소수집 원칙에 따라 재검토합니다.
- 키/몸무게, 가족력, 흡연/음주/운동, 혈압, 혈당, 지질 수치는 별도 건강정보 입력 화면에서 받습니다.
- 휴대폰 SMS 인증은 제외 항목이 아니며 Twilio Verify 기반으로 구현합니다. 로컬/개발 환경은 no-op 또는 개발용 인증번호 흐름을 사용할 수 있고, 운영에서는 debug 값을 응답하지 않습니다.
- `users.role`은 권한 판단의 기준으로 사용합니다.
- `is_admin`은 legacy 호환 필드로 남기고, 신규 관리자 권한 체크는 `role` 기반으로 통일합니다.
- 관리자 role은 `USER`, `MONITOR`, `OPERATOR`, `ADMIN`, `SUPER_ADMIN` 구조로 설계합니다.
- 관리자 API는 최소 권한에 따라 `require_monitor_user`, `require_operator_user`, `require_admin_user`, `require_super_admin_user`를 사용합니다.
- `MONITOR`는 읽기 전용 운영 관찰자, `OPERATOR`는 문의/FAQ/챌린지/알림 운영 담당, `ADMIN`은 사용자/서비스 관리 담당, `SUPER_ADMIN`은 권한 관리/위험 설정/모델 활성화 담당으로 분리합니다.
- 로그인 실패 횟수는 `failed_login_count`로 기록하고, 5회 이상 실패 시 CAPTCHA 등 추가 확인을 요구하는 soft-lock 구조로 설계합니다. CAPTCHA 도입 전에는 짧은 제한과 일반화된 안내 메시지를 사용하고, 성공 로그인 시 실패 횟수와 제한 상태를 초기화합니다.
- 로그인 시각 표준 필드는 `last_login_at`입니다. `last_login`은 legacy 호환 필드로 남기되 신규 갱신 기준으로 사용하지 않습니다.
- `health_records`의 흡연/음주/운동 관련 중복 컬럼은 x1/x2 전환기 호환 필드로 보고, 실제 모델 입력 스키마 확정 후 별도 schema cleanup에서 정리합니다.
- `medication_records.user_id`는 조회 최적화와 호환성 검토 후 별도 제거 여부를 판단합니다.
- 비밀번호 해싱은 bcrypt 호환 없이 Argon2id 단일 방식으로 전환합니다.
- 기존 로컬 개발 계정에 남아 있는 예전 해시는 로그인되지 않을 수 있으므로 재가입하거나 비밀번호 재설정 후 사용합니다.
- 운영 전환 시에는 기존 계정의 비밀번호 재설정 또는 별도 전환 정책을 사전에 수립해야 합니다.
- `/api/v1/system/health`는 API alive, database, Redis 상태를 함께 반환합니다.
- 모든 HTTP 응답에는 `X-Request-ID`를 포함하고, 요청 처리 중 `request.state.request_id`로 추적할 수 있습니다.
- 소셜 로그인과 웨어러블 연동은 이번 1차 범위에서 구현하지 않습니다.
- 향후 소셜 로그인은 카카오/네이버를 우선 검토하고, 애플은 후순위로 둡니다.
- 정확한 영양성분/그램 단위 계산은 후순위로 두고, 초기 식단 기능은 기록, 이미지 기반 간편 분석, 사용자 보정 중심으로 설계합니다.
- 가족공유는 후속 스프린트 우선순위로 두되, 풀서비스 범위에서는 구현 대상으로 유지합니다.
- AI Worker, `async_jobs`, Redis queue 연결은 실제 ML/CV/LLM 운영 연동 단계의 후순위 작업으로 둡니다. 현재 P0 인증/계정 정책 정리 범위에서는 구현하지 않습니다.
- SMTP/SES/Solapi 등 실제 이메일 발송은 후속 구현 대상입니다. 운영환경에서는 이메일 인증코드와 비밀번호 재설정 토큰을 응답 본문에 포함하지 않습니다.
- DB/migration 변경은 각 기능 구현 단계에서 별도 설계와 리뷰를 거쳐 진행합니다.
- 의료 분석 결과는 진단/처방이 아니라 건강관리 참고 정보라는 정책을 UI/API/문서에 일관되게 반영합니다.
