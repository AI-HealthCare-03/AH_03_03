# 관리자 접근/권한 설계

이 문서는 풀서비스 기준 관리자 콘솔 접근 구조와 role 정책을 정의한다.
이번 설계는 코드 구현이 아니라 향후 구현 기준이다.

## 1. 관리자 권한 체계 개요

헬스케어 서비스에서는 일반적인 `USER`/`ADMIN` 2단계 권한만으로 운영 책임을 안전하게 나누기 어렵다.
관리자는 사용자 계정, 건강정보, 검진표, 복약정보, 분석 결과, OCR 원본, 알림, 모델 설정처럼 민감하고 영향도가 큰 리소스에 접근할 수 있기 때문이다.

권한을 세분화하는 이유:

- 모니터링 담당자는 운영 상태를 봐야 하지만 사용자 정보를 수정하면 안 된다.
- 고객지원/콘텐츠 운영자는 문의/FAQ/챌린지는 처리할 수 있지만 관리자 권한을 바꾸면 안 된다.
- 사용자/서비스 관리자는 계정 상태를 관리할 수 있지만 최고 권한 부여나 위험 설정 변경은 제한해야 한다.
- 최고관리자는 모델 활성화, 위험 설정, 관리자 권한 회수 같은 고위험 작업을 담당한다.
- 민감정보 조회는 역할이 충분해도 audit log가 필요하다.

## 2. Role 기준 정책

권한 판단의 진실 공급원은 `users.role`이다.

`is_admin`은 legacy 호환 필드로만 남긴다.
신규 권한 판단, dependency, admin route 보호에서 `is_admin`을 사용하지 않는다.
`role`과 `is_admin` 값이 충돌하면 항상 `users.role`을 우선한다.

기본 role:

- `USER`: 일반 사용자
- `MONITOR`: 모니터링 관리자, 읽기 전용 운영 관찰자
- `OPERATOR`: 중간관리자, 문의/FAQ/챌린지/알림 운영 담당
- `ADMIN`: 사용자/서비스 관리 담당
- `SUPER_ADMIN`: 최고관리자, 권한 관리/위험 설정/모델 활성화 담당

## 3. Role별 권한 정의

### USER

권한:

- 본인 데이터만 접근
- 본인 건강정보, 분석 결과, 챌린지, 식단, 복약, 알림 사용
- 본인 계정 정보 수정

제한:

- 관리자 콘솔 접근 불가
- 타 사용자 데이터 접근 불가

### MONITOR

읽기 전용 운영 관찰자다.

가능:

- 운영 대시보드 조회
- 시스템 상태 조회
- 성능 지표 조회
- 에러/실패 현황 조회
- 분석/OCR/알림 처리 현황 조회
- Worker/Queue 상태 조회

불가:

- 사용자 정보 수정
- 건강정보 수정/삭제
- FAQ/문의 답변 작성
- 챌린지 생성/수정
- 알림 수동 발송
- 관리자 권한 변경

### OPERATOR

고객지원과 콘텐츠/운영 처리를 담당하는 중간관리자다.

가능:

- 문의 답변
- FAQ 관리
- 챌린지 콘텐츠 관리
- 알림 템플릿/운영 알림 관리
- OCR/분석 실패 케이스 확인
- 운영성 실패 케이스 확인

제한:

- 사용자 계정 삭제 불가
- 관리자 권한 변경 불가
- 민감 건강정보 원문 조회는 제한 또는 audit log 필수
- 모델 활성화/위험 설정 변경 불가

### ADMIN

사용자와 서비스 운영을 관리한다.

가능:

- 사용자 목록/상세 관리
- 사용자 상태 관리
- 문의/FAQ/챌린지/알림 운영 관리
- 분석/OCR/식단/복약 운영 데이터 확인
- 일부 민감정보 조회. 단 audit log 필수

제한:

- `SUPER_ADMIN` 권한 부여 불가
- 최고 위험 설정 변경 불가
- 활성 모델 버전 변경은 정책 결정 필요

### SUPER_ADMIN

최고관리자다.

가능:

- 관리자 권한 부여/회수
- 시스템 위험 설정 변경
- 활성 모델 버전 변경
- 사용자 강제 비활성화/삭제
- 보안/운영 핵심 설정 변경
- 모든 관리자 활동 로그 조회
- 전체 audit log 조회

## 4. Role별 접근 화면

| 화면 | 최소 role | 비고 |
|---|---|---|
| `/admin` | MONITOR | 운영 요약 대시보드 |
| `/admin/monitoring` | MONITOR | 시스템/성능/실패 현황 읽기 전용 |
| `/admin/system` | MONITOR | health, worker, queue 상태 |
| `/admin/analysis` | MONITOR | 분석 처리 현황. 민감 상세는 audit log |
| `/admin/ocr` | MONITOR | OCR 처리 현황. 원본 파일 조회는 ADMIN 이상 권장 |
| `/admin/diets` | MONITOR | 식단 분석 현황 |
| `/admin/notifications` | MONITOR | 발송 로그 조회는 MONITOR, 템플릿 관리는 OPERATOR |
| `/admin/inquiries` | OPERATOR | 문의 답변 |
| `/admin/faqs` | OPERATOR | FAQ 생성/수정/삭제 |
| `/admin/challenges` | OPERATOR | 챌린지 콘텐츠 관리 |
| `/admin/users` | ADMIN | 사용자 목록/상세/상태 관리 |
| `/admin/models` | SUPER_ADMIN | 활성 모델 변경은 SUPER_ADMIN 기본. ADMIN 조회 권한은 별도 정책 |
| `/admin/audit-logs` | SUPER_ADMIN | ADMIN 일부 조회 가능 여부는 별도 정책 |
| `/admin/settings` | SUPER_ADMIN | 보안/운영 핵심 설정 |

## 5. Role별 API 설계 기준

| API | 최소 role | 비고 |
|---|---|---|
| `GET /admin/summary` | MONITOR | 운영 요약 |
| `GET /admin/system/health` | MONITOR | deep health/readiness |
| `GET /admin/system/errors` | MONITOR | system error logs |
| `GET /admin/analysis-jobs` | MONITOR | 분석 처리 현황 |
| `GET /admin/ocr-jobs` | MONITOR | OCR 처리 현황 |
| `GET /admin/notification-logs` | MONITOR | 알림 발송 로그 |
| `GET /admin/users` | ADMIN | 사용자 목록 |
| `GET /admin/users/{id}` | ADMIN | 민감정보 접근 시 audit log |
| `PATCH /admin/users/{id}/status` | ADMIN | 활성/비활성/잠금 |
| `POST /admin/inquiries/{id}/answer` | OPERATOR | 문의 답변 |
| `CRUD /admin/faqs` | OPERATOR | FAQ 운영 |
| `CRUD /admin/challenges` | OPERATOR | 챌린지 운영 |
| `CRUD /admin/notification-templates` | OPERATOR | 알림 템플릿 |
| `GET /admin/models` | ADMIN | 모델 목록 조회 |
| `PATCH /admin/models/{id}/activate` | SUPER_ADMIN | 활성 모델 변경 |
| `GET /admin/audit-logs` | SUPER_ADMIN | ADMIN 제한 조회 가능성 검토 |
| `PATCH /admin/roles/{user_id}` | SUPER_ADMIN | 관리자 권한 부여/회수 |
| `PATCH /admin/settings/risk-thresholds` | SUPER_ADMIN | 위험 설정 변경 |

## 6. Dependency 설계

추후 코드 구현 시 아래 dependency를 만든다.

| dependency | 허용 role |
|---|---|
| `require_monitor_user` | MONITOR, OPERATOR, ADMIN, SUPER_ADMIN |
| `require_operator_user` | OPERATOR, ADMIN, SUPER_ADMIN |
| `require_admin_user` | ADMIN, SUPER_ADMIN |
| `require_super_admin_user` | SUPER_ADMIN |

공통 정책:

- 현재 로그인 사용자가 필요하다.
- `users.role`만 기준으로 판단한다.
- 권한이 부족하면 403을 반환한다.
- 프론트에서 메뉴를 숨기더라도 백엔드 dependency 검사는 반드시 적용한다.

## 7. 관리자 콘솔 분리 정책

개발 단계:

- `/admin/login`
- `/admin/*`
- 기존 React 앱 안에 `AdminRoute`와 `AdminLayout`으로 시작 가능

운영 단계:

- 사용자 앱: `app.service.com`
- 관리자 콘솔: `admin.service.com`
- API 서버: `api.service.com`

관리자 콘솔 추가 보안:

- 2FA
- IP allowlist
- rate limiting
- 관리자 로그인 알림
- 짧은 세션 만료
- audit log
- 민감정보 조회 사유 입력

관리자 페이지가 일반 운영 사이트에 그대로 노출될 때 위험:

- brute force
- 관리자 URL 스캔
- 일반 사용자와 UI 혼동
- 민감정보 접근 위험
- 권한 우회 시도

## 8. Audit Log 정책

아래 행동은 반드시 audit log 대상이다.

- 사용자 상세 조회
- 사용자 건강정보 조회
- 검진표/OCR 원본 조회
- 분석 결과 상세 조회
- 복약정보 조회
- 사용자 상태 변경
- 회원탈퇴/비활성화 처리
- FAQ 생성/수정/삭제
- 문의 답변
- 챌린지 생성/수정/삭제
- 알림 수동 발송
- 모델 활성 버전 변경
- 관리자 권한 부여/회수
- 시스템 설정 변경

권장 필드:

- `id`
- `actor_user_id`
- `actor_role`
- `target_user_id`
- `action_type`
- `resource_type`
- `resource_id`
- `ip_address`
- `user_agent`
- `request_id`
- `created_at`

## 9. Permission Table 확장 가능성

초기에는 `users.role` enum 성격의 문자열로 시작한다.
역할이 더 세분화되면 `admin_permissions` 테이블을 추가할 수 있다.

예시:

- `id`
- `user_id`
- `permission_key`
- `granted_by`
- `created_at`
- `revoked_at`

`permission_key` 예:

- `VIEW_SYSTEM_METRICS`
- `VIEW_AUDIT_LOGS`
- `MANAGE_USERS`
- `MANAGE_FAQS`
- `ANSWER_INQUIRIES`
- `MANAGE_CHALLENGES`
- `MANAGE_MODELS`
- `MANAGE_NOTIFICATIONS`

## 10. 구현 우선순위

### P0

- role enum 정책 확정
- `require_*` dependency 구현
- `AdminRoute` / `AdminLayout` 설계
- `audit_logs` 기본 설계
- `/admin/monitoring` 읽기 전용 화면

### P1

- 사용자 관리
- 문의/FAQ 관리
- 챌린지 관리
- `notification_logs`
- `system_error_logs`
- `request_id` 기반 추적

### P2

- 관리자 role 세분화
- permission table
- 2FA
- IP allowlist
- admin 서브도메인 분리
- 모델 관리
- 비용/성능 대시보드

## 11. 기존 코드 영향 범위

추후 구현 시 영향 받을 영역:

- `app/models/users.py`
- `app/apis/v1/dependencies.py`
- `app/apis/v1/admin_routers.py`
- `app/services/admin.py`
- `frontend/src/pages/admin/*`
- `frontend/src/api/admin.ts`
- `frontend/src/auth/AdminRoute.tsx`
- `frontend/src/components/AdminLayout.tsx`
- `audit_logs` migration
- `system_error_logs` migration

## 12. 금지/주의사항

- 프론트에서 관리자 메뉴를 숨기는 것만으로는 보안이 아니다.
- 모든 관리자 API는 백엔드에서 role을 확인해야 한다.
- 관리자 기능을 일반 사용자 사이드바에 노출하지 않는다.
- 민감정보 조회 로그 없이 관리자 조회를 허용하지 않는다.
- `is_admin`을 신규 권한 판단 기준으로 사용하지 않는다.
- Firebase/social login은 1차 범위 제외 상태를 유지한다.
