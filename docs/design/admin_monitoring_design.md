# 관리자/운영 모니터링 시스템 설계

## 1. 전체 개요

관리자/운영 모니터링 시스템은 풀서비스 운영에서 사용자 지원, 장애 대응, AI 처리 품질 확인, 민감정보 접근 통제, 운영 감사 추적을 담당하는 내부 관리 기능이다.

MVP에서는 프론트 시연과 기본 API 흐름 확인이 중심이므로 관리자 화면, worker 모니터링, 모델 관리, 발송 로그, 감사 로그가 최소화되어 있다. 풀서비스에서는 실제 사용자 건강정보, 검진표 원본, 식단 사진, 복약 정보, AI 분석 결과를 다루기 때문에 운영자 접근 권한과 감사 로그가 필수다.

운영 로그와 감사 로그가 중요한 이유:

- 헬스케어 데이터는 민감정보이므로 누가, 언제, 어떤 목적으로 조회했는지 추적해야 한다.
- 분석/OCR/LLM/알림 실패를 운영자가 빠르게 확인하고 재처리해야 한다.
- 모델 버전 변경, 임계값 변경, 고위험 알림 정책 변경은 서비스 결과에 직접 영향을 준다.
- 관리자 오조작, 과도한 개인정보 조회, 권한 남용을 방지해야 한다.
- 장애 대응과 고객 문의 대응에 필요한 이력을 안전하게 보관해야 한다.

## 2. 관리자 권한 모델

관리자 role은 `USER`, `MONITOR`, `OPERATOR`, `ADMIN`, `SUPER_ADMIN` 5단계로 설계한다.

설계 원칙:

- `users.role`은 시스템 권한의 진실 공급원이다.
- `is_admin`은 legacy 호환 필드이며 신규 권한 판단 기준이 아니다.
- role 충돌 시 항상 `users.role`을 우선한다.
- `MONITOR`는 읽기 전용 운영 관찰자다.
- `OPERATOR`는 문의/FAQ/챌린지/알림 콘텐츠를 처리하는 중간관리자다.
- `ADMIN`은 사용자/서비스 관리 담당이다.
- `SUPER_ADMIN`은 관리자 권한, 위험 설정, 모델 활성화 같은 최고 권한을 가진다.
- 민감정보 상세 조회는 역할이 충분해도 audit log를 남긴다.

| role | 요약 | 주요 권한 |
|---|---|---|
| `USER` | 일반 사용자 | 본인 데이터만 접근 |
| `MONITOR` | 모니터링 관리자 | 운영 대시보드, 성능, 실패 현황 읽기 전용 |
| `OPERATOR` | 중간관리자 | 문의 답변, FAQ/챌린지/알림 템플릿 운영 |
| `ADMIN` | 서비스 관리자 | 사용자 관리, 운영 데이터 확인, 일부 민감정보 조회 |
| `SUPER_ADMIN` | 최고관리자 | 관리자 권한 부여/회수, 모델 활성화, 위험 설정 변경 |

권한 포함 관계:

- `require_monitor_user`: `MONITOR`, `OPERATOR`, `ADMIN`, `SUPER_ADMIN`
- `require_operator_user`: `OPERATOR`, `ADMIN`, `SUPER_ADMIN`
- `require_admin_user`: `ADMIN`, `SUPER_ADMIN`
- `require_super_admin_user`: `SUPER_ADMIN`

2026-05-23 1차 구현 상태:

- 백엔드 공통 dependency 4종을 구현했다.
- 기존 `is_admin` 필드는 유지하지만 신규 권한 판단에는 사용하지 않는다.
- FAQ/문의/챌린지 운영성 쓰기 API는 `OPERATOR` 이상으로 보호한다.
- LLM log 조회는 `MONITOR` 이상으로 보호한다.
- 관리자 콘솔, audit log, 별도 admin router는 아직 구현하지 않았다.
- `X-Request-ID` 기반 요청 추적과 `system_error_logs` 500 서버 예외 저장은 P0 기반으로 1차 구현했다.
- `sensitive_access_logs`는 건강정보/분석결과/검진표/복약정보/대시보드 조회 접근 기록용으로 1차 구현했다. 관리자 콘솔 조회 UI는 후속 작업이다.

## 3. 관리자 주요 화면

권장 라우트:

- `/admin`: 운영 대시보드
- `/admin/monitoring`: 읽기 전용 운영 모니터링
- `/admin/users`: 사용자 목록/검색
- `/admin/users/:id`: 사용자 상세
- `/admin/inquiries`: 문의 관리
- `/admin/faqs`: FAQ 관리
- `/admin/analysis`: 분석 모니터링
- `/admin/ocr`: OCR 모니터링
- `/admin/diets`: 식단 분석 모니터링
- `/admin/notifications`: 알림/리마인드 모니터링
- `/admin/models`: 모델 관리
- `/admin/system`: 시스템/Worker 상태
- `/admin/audit-logs`: 감사 로그 조회
- `/admin/settings`: 관리자/운영 설정

화면별 최소 role:

| 화면 | 최소 role |
|---|---|
| `/admin`, `/admin/monitoring`, `/admin/system` | MONITOR |
| `/admin/analysis`, `/admin/ocr`, `/admin/diets`, `/admin/notifications` 조회 | MONITOR |
| `/admin/inquiries`, `/admin/faqs`, `/admin/challenges` | OPERATOR |
| `/admin/users` | ADMIN |
| `/admin/models` | SUPER_ADMIN 기본. 조회 권한은 ADMIN 허용 여부 검토 |
| `/admin/audit-logs` | SUPER_ADMIN 기본. ADMIN 제한 조회 여부 검토 |
| `/admin/settings` | SUPER_ADMIN |

화면 공통 요구:

- 관리자 전용 layout과 일반 사용자 layout 분리
- 위험 작업은 confirm 또는 재인증 필요
- 민감정보 상세 버튼은 명확한 목적 선택 후 접근
- 목록에는 민감 원문 대신 요약/마스킹 정보 우선 표시

## 4. 운영 대시보드 지표

`GET /admin/summary`에서 제공할 수 있는 핵심 지표:

- 오늘 가입자 수
- 활성 사용자 수
- 분석 요청 수
- OCR 처리 수
- 식단 분석 수
- 챗봇 질문 수
- 알림 발송 수
- 실패 job 수
- 미답변 문의 수
- 고위험 분석 결과 수
- 평균 API 응답 시간
- worker queue backlog
- LLM/OCR 예상 비용

권장 표시:

- 오늘/최근 7일/최근 30일 필터
- 성공/실패 추이
- 실패율 warning threshold
- 고위험 결과는 개인정보 없이 count와 추세만 표시

## 5. 사용자 관리

### 5.1 사용자 목록

표시 항목:

- 사용자 id
- 이메일/아이디 마스킹
- 이름 마스킹
- 가입일
- 최근 로그인
- 활성/비활성 상태
- 탈퇴 요청/비활성화 상태
- 건강정보 입력 여부
- 분석 횟수

기능:

- 검색: 이메일, login_id, 이름, 휴대폰. 결과는 마스킹.
- 필터: 활성/비활성, 가입일, 최근 로그인, 분석 여부.
- 상세 이동.

### 5.2 사용자 상세

표시 항목:

- 기본 계정 정보
- 최근 로그인
- 건강정보 입력 여부
- 최근 분석 요약
- 최근 OCR/식단/복약/챌린지 활동 요약
- 문의 내역
- 알림 수신 상태

민감정보 정책:

- 건강 수치 상세, 검진표 원본, 식단 사진, 복약 상세를 열람하면 `audit_logs`에 기록한다.
- 운영 편의를 위해 전체 원본 데이터를 기본 노출하지 않는다.
- 필요 시 “민감정보 조회 사유” 입력 후 상세 조회한다.

### 5.3 사용자 상태 변경

가능 작업:

- 계정 활성/비활성
- 탈퇴 상태 확인
- 임시 잠금
- 관리자 메모

삭제 정책:

- 물리 삭제보다 비활성화/탈퇴 처리 우선.
- 개인정보 삭제/익명화는 별도 정책과 배치 작업 필요.

## 6. 문의/FAQ 관리

### 6.1 문의 관리

기능:

- 문의 목록 조회
- 카테고리/상태/작성일 필터
- 문의 상세 조회
- 답변 등록/수정
- 답변 완료 처리
- 담당자 메모

상태 예:

- `PENDING`
- `IN_PROGRESS`
- `ANSWERED`
- `CLOSED`

### 6.2 FAQ 관리

기능:

- FAQ 목록 조회
- FAQ 생성/수정/삭제
- FAQ 활성/비활성
- 카테고리 관리
- 표시 순서 변경
- 검색 키워드 관리

권한:

- `ADMIN` 이상 가능
- 추후 `CONTENT_MANAGER`에게 위임 가능

## 7. 분석 모니터링

대상:

- 건강 위험도 분석 요청
- 질환별 결과
- feature factor
- snapshot
- 모델 버전
- 실패 이력

목록 표시:

- 요청 id
- 사용자 id
- 분석 타입
- 상태
- risk level
- model version
- 요청 시각/완료 시각
- 오류 메시지

상세 표시:

- 입력 snapshot 요약
- 모델 출력
- rule output
- factor 목록
- 처리 시간
- 재처리 가능 여부

재실행/재처리 정책:

- 같은 입력 snapshot 기준으로 재실행한다.
- 재실행 시 기존 결과를 덮어쓰지 않고 새 결과로 저장하는 것을 권장한다.
- 재처리 이력은 `admin_activity_logs`와 `ai_inference_logs`에 남긴다.

## 8. OCR 모니터링

대상:

- 검진표 OCR
- 복약/처방전 OCR
- OCR 실패
- confidence 낮은 결과
- 사용자 수정 여부

목록 표시:

- OCR job id
- 사용자 id
- OCR 유형
- 상태
- 평균 confidence
- 사용자 확인 여부
- 원본 파일 존재 여부
- 처리 시간
- 실패 메시지

상세 표시:

- 추출 항목 목록
- confidence
- 사용자 수정 전/후 값
- 원본 파일 접근 버튼

원본 파일 접근 권한:

- 원본 검진표/처방전/약봉투 이미지는 민감정보다.
- 원본 파일 조회는 `ADMIN` 이상이라도 사유 입력과 audit log가 필요하다.
- 임시 signed URL을 사용하고 만료 시간을 짧게 둔다.

재처리 정책:

- OCR engine version과 prompt/template version을 기록한다.
- 재처리 시 기존 결과와 새 결과를 비교할 수 있게 보관한다.

## 9. 식단 분석 모니터링

대상:

- 식단 이미지 분석 요청
- 식단 점수
- 인식 음식명
- 영양성분 요약
- 사용자 보정 이력
- 실패/재처리

목록 표시:

- diet record id
- 사용자 id
- meal type
- score
- analysis method
- 상태
- 생성일
- 사용자 보정 여부

상세 표시:

- 인식 음식명
- nutrition summary
- diet feedback
- 사용자 보정 전/후
- 이미지 접근 여부

주의:

- 식단 사진도 민감정보로 취급한다.
- 원본 이미지 접근은 audit log 대상이다.

## 10. 알림 모니터링

관리 대상:

- `reminder_schedules`
- `notification_logs`
- 발송 성공/실패
- 가족 알림
- 복약 알림
- 챌린지 알림
- 고위험 수치 알림

목록 표시:

- 알림 id
- 사용자 id
- notification type
- channel
- status
- scheduled_at
- sent_at
- failure reason
- retry count

정책:

- 앱 내부 알림과 외부 발송 로그를 분리한다.
- 외부 Push/SMS/Email/Kakao 발송은 provider 응답과 비용을 기록한다.
- 실패 알림은 재시도 정책을 가진다.
- 가족 알림은 공유 권한을 재확인한 뒤 발송한다.

## 11. 모델 관리

대상 테이블:

- `model_versions`
- `model_thresholds`

관리 기능:

- 모델 버전 목록
- 활성 모델 확인
- 모델 성능 메트릭 확인
- 질환별 threshold 확인/수정
- 모델 활성화
- rollback
- 모델 배포 이력

권한:

- 조회: `ADMIN`
- 활성화/rollback/threshold 수정: `SUPER_ADMIN`
- 추후 `MEDICAL_REVIEWER` 검수 승인 플로우 가능

필수 로그:

- 모델 활성화
- threshold 변경
- rollback
- 모델 파일 등록
- 성능 지표 수정

## 12. 시스템/Worker 모니터링

관리 대상:

- `async_jobs`
- worker 상태
- queue backlog
- job 실패율
- LLM/OCR 비용
- API error rate

화면 구성:

- FastAPI health
- PostgreSQL health
- Redis health
- AI Worker heartbeat
- Notification Worker heartbeat
- Report Worker heartbeat
- Queue depth
- 최근 실패 job
- 비용 지표

worker heartbeat:

- worker id
- worker type
- last_seen_at
- current_job_id
- status
- version

장애 대응:

- 실패 job 재시도
- 특정 job 취소
- worker 재시작은 인프라 권한 필요
- 장애 타임라인 기록

## 13. 필요한 DB 테이블

### 13.1 `audit_logs`

민감정보 접근과 주요 보안 이벤트 감사 로그.

권장 컬럼:

- `id BIGSERIAL PK`
- `actor_user_id BIGINT FK users.id ON DELETE SET NULL`
- `actor_role VARCHAR(30)`
- `action_type VARCHAR(80) NOT NULL`
- `target_type VARCHAR(80)`
- `target_id BIGINT`
- `target_user_id BIGINT FK users.id ON DELETE SET NULL`
- `reason TEXT`
- `metadata JSONB`
- `ip_address VARCHAR(45)`
- `user_agent TEXT`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`

### 13.2 `admin_activity_logs`

관리자 조작 이력.

예:

- 사용자 비활성화
- FAQ 수정
- 문의 답변
- 모델 활성화
- job 재처리

### 13.3 `async_jobs`

AI/Worker 비동기 처리 job.

대상:

- health analysis
- OCR
- diet analysis
- chatbot/RAG preprocessing
- notification dispatch
- report generation

### 13.4 `system_event_logs`

시스템 장애, 배치 실패, 외부 provider 장애 기록.

### 13.5 `ai_inference_logs`

모델 추론 로그.

권장 정보:

- model_version_id
- analysis_result_id
- input_schema_version
- inference latency
- model output
- threshold
- status/error

### 13.6 `ocr_processing_logs`

OCR 처리 로그.

권장 정보:

- ocr type
- provider
- template version
- confidence summary
- raw provider status
- error message

### 13.7 `notification_logs`

외부 알림 발송 로그.

권장 정보:

- notification id
- channel
- provider
- recipient hash
- status
- provider response
- retry count

### 13.8 `reminder_schedules`

복약/챌린지/식단/건강기록 리마인드 스케줄.

### 13.9 `file_uploads`

검진표, 식단 사진, 처방전, 약봉투 파일 메타데이터.

권장 정보:

- owner_user_id
- file_type
- storage_key
- original_filename
- content_type
- size
- checksum
- scan_status
- created_at

### 13.10 `model_versions`

모델 버전 registry.

권장 정보:

- model_name
- disease_type
- version
- artifact_path
- feature_schema_version
- metrics
- is_active
- activated_at

### 13.11 `model_thresholds`

질환별 threshold 관리.

권장 정보:

- model_version_id
- risk_level
- min_score
- max_score
- description

## 14. 필요한 API

### 14.1 운영 요약

- `GET /admin/summary`
- `GET /admin/system/health`

최소 role: `MONITOR`

### 14.2 사용자 관리

- `GET /admin/users`
- `GET /admin/users/{id}`
- `PATCH /admin/users/{id}/status`
- `GET /admin/users/{id}/activity-summary`
- `GET /admin/users/{id}/sensitive-access-summary`

최소 role: `ADMIN`

### 14.3 문의/FAQ

- `GET /admin/inquiries`
- `GET /admin/inquiries/{id}`
- `POST /admin/inquiries/{id}/answer`
- `GET /admin/faqs`
- `POST /admin/faqs`
- `PATCH /admin/faqs/{id}`
- `DELETE /admin/faqs/{id}`

최소 role: `OPERATOR`

### 14.4 분석/OCR/식단

- `GET /admin/analysis-jobs`
- `GET /admin/analysis-jobs/{id}`
- `POST /admin/analysis-jobs/{id}/retry`
- `GET /admin/ocr-jobs`
- `GET /admin/ocr-jobs/{id}`
- `POST /admin/ocr-jobs/{id}/retry`
- `GET /admin/diet-jobs`
- `GET /admin/diet-jobs/{id}`

조회 최소 role: `MONITOR`
재처리 최소 role: `OPERATOR` 이상 또는 정책 결정

### 14.5 알림

- `GET /admin/notification-logs`
- `GET /admin/notification-logs/{id}`
- `POST /admin/notification-logs/{id}/retry`
- `GET /admin/reminder-schedules`
- `PATCH /admin/reminder-schedules/{id}`

로그 조회 최소 role: `MONITOR`
템플릿/스케줄 수정 최소 role: `OPERATOR`

### 14.6 모델

- `GET /admin/models`
- `GET /admin/models/{id}`
- `PATCH /admin/models/{id}/activate`
- `POST /admin/models/{id}/rollback`
- `GET /admin/models/{id}/thresholds`
- `PATCH /admin/models/{id}/thresholds`

조회 최소 role: `ADMIN`
활성화/rollback/threshold 변경 최소 role: `SUPER_ADMIN`

### 14.7 감사 로그

- `GET /admin/audit-logs`
- `GET /admin/admin-activity-logs`

최소 role: `SUPER_ADMIN`
일부 감사 로그는 `ADMIN` 제한 조회를 검토할 수 있다.

## 15. 보안 정책

### 15.1 관리자 접근

- 관리자 페이지는 화면/API별 최소 role을 적용한다.
- 읽기 전용 모니터링은 `MONITOR` 이상에게 허용한다.
- 운영 처리 기능은 `OPERATOR` 이상에게 허용한다.
- 사용자 관리 기능은 `ADMIN` 이상에게 허용한다.
- `SUPER_ADMIN` 기능은 route, service, UI에서 명확히 분리한다.
- 프론트에서 관리자 메뉴를 숨기는 것만으로는 보안이 아니며 모든 관리자 API는 백엔드 role 검사를 수행한다.
- 신규 권한 판단에서 `is_admin`을 사용하지 않는다.

### 15.2 민감정보 접근

민감정보 예:

- 건강 수치 상세
- 분석 입력 snapshot
- 검진표 원본
- 식단 사진
- 처방전/약봉투 원본
- 복약 상세
- 가족 공유 상세

정책:

- 민감정보 조회 시 `audit_logs` 저장.
- 조회 사유 입력 권장.
- 관리자 화면 기본 목록에서는 원문을 노출하지 않는다.
- 원본 파일은 짧은 만료 시간을 가진 signed URL로 접근한다.

### 15.3 관리자 활동 로그

저장 대상:

- 사용자 상세 조회
- 사용자 건강정보 조회
- 검진표/OCR 원본 조회
- 분석 결과 상세 조회
- 복약정보 조회
- 사용자 상태 변경
- 회원탈퇴/비활성화 처리
- 문의 답변
- FAQ 생성/수정/삭제
- 챌린지 생성/수정/삭제
- 알림 수동 발송
- job 재처리
- 모델 활성 버전 변경
- threshold 변경
- 관리자 권한 부여/회수
- 시스템 설정 변경

기록 필드:

- actor_user_id
- actor_role
- target_user_id
- action_type
- resource_type
- resource_id
- before/after metadata
- ip_address
- user_agent
- request_id
- created_at

### 15.4 삭제/비활성화 정책

- 삭제/비활성화는 confirm 필요.
- 대량 작업은 2단계 confirm 또는 SUPER_ADMIN 승인.
- 운영 DB seed 금지.
- 운영 DB에서 임의 test user 생성 금지.

### 15.5 기타 보안

- 관리자 API rate limit
- 관리자 세션 만료 시간 단축
- 관리자 계정 2FA 검토
- CORS 운영 origin 제한
- 파일 업로드 malware scan
- XSS 방지를 위한 관리자 입력 sanitization

## 16. 구현 단계

### P0: 관리자 최소 운영 기능

- role enum 정책 확정
- `require_monitor_user`, `require_operator_user`, `require_admin_user`, `require_super_admin_user`
- `AdminRoute` / `AdminLayout` 설계
- `/admin/monitoring` 읽기 전용 화면
- `audit_logs`
- `admin_activity_logs`
- 민감정보 조회 audit helper

### P1: 서비스 모니터링 확장

- 사용자 관리
- 문의/FAQ 관리
- 챌린지 관리
- 분석 job 모니터링
- OCR job 모니터링
- 식단 분석 모니터링
- 알림 로그 모니터링
- `async_jobs`
- `notification_logs`
- `reminder_schedules`
- `file_uploads`
- 실패 job 재처리 API

### P2: 모델/Worker/비용 관리

- `model_versions`
- `model_thresholds`
- 모델 활성화/rollback
- worker heartbeat
- queue backlog
- LLM/OCR 비용 지표
- API error rate
- 관리자 권한 세분화
- permission table
- 관리자 2FA
- IP allowlist
- admin 서브도메인 분리

## 17. 기존 코드 영향 범위

### Backend

예상 추가/수정:

- `app/apis/v1/admin_routers.py`
- `app/services/admin.py`
- `app/repositories/admin_repository.py`
- `app/models/admin.py` 또는 `app/models/logs.py`
- `app/dtos/admin.py`
- `app/apis/v1/dependencies.py`: `ensure_super_admin_user` 추가 가능
- `app/apis/v1/dependencies.py`: `require_monitor_user`, `require_operator_user`, `require_admin_user`, `require_super_admin_user`
- `app/apis/v1/__init__.py`: admin router 등록
- `app/models/users.py`: role 값 확장 검토

### Frontend

예상 추가/수정:

- `frontend/src/api/admin.ts`
- `frontend/src/pages/admin/AdminDashboardPage.tsx`
- `frontend/src/pages/admin/AdminUsersPage.tsx`
- `frontend/src/pages/admin/AdminUserDetailPage.tsx`
- `frontend/src/pages/admin/AdminInquiriesPage.tsx`
- `frontend/src/pages/admin/AdminFaqsPage.tsx`
- `frontend/src/pages/admin/AdminAnalysisPage.tsx`
- `frontend/src/pages/admin/AdminOcrPage.tsx`
- `frontend/src/pages/admin/AdminDietsPage.tsx`
- `frontend/src/pages/admin/AdminNotificationsPage.tsx`
- `frontend/src/pages/admin/AdminModelsPage.tsx`
- `frontend/src/pages/admin/AdminSystemPage.tsx`
- `frontend/src/pages/admin/AdminAuditLogsPage.tsx`
- `frontend/src/components/admin/AdminLayout.tsx`
- 일반 사용자 `Sidebar`와 관리자 `AdminLayout` 분리

### Worker/Infra

예상 추가:

- worker heartbeat
- async job retry
- queue depth metric
- structured logging
- external provider cost aggregation
- monitoring dashboard integration

## 18. 추후 고려 사항

- 관리자 계정 2FA
- 관리자 IP allowlist
- 관리자 권한 승인 workflow
- 민감정보 조회 사유 필수화
- audit log 보관 기간과 파기 정책
- 로그 위변조 방지
- SIEM 연동
- 장애 알림 Slack/Email 연동
- 개인정보 export/delete 요청 처리
- 관리자 작업 replay 방지
- 모델 변경 승인 절차
- 의료/법무 검수 workflow
