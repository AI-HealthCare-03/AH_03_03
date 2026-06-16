# 가족 관리 서비스 설계

## 1. 전체 개요

가족 관리 기능은 사용자가 본인의 건강정보를 가족 또는 보호자와 안전하게 공유하고, 가족 구성원의 건강 이상 징후를 함께 확인할 수 있게 하는 풀서비스 확장 기능이다.

핵심 원칙은 다음과 같다.

- `users.role`은 `USER`, `ADMIN` 같은 시스템 권한만 표현한다.
- 가족 관계 안에서의 역할은 `family_members.member_role`로 표현한다.
- 한 사용자는 본인의 건강관리 사용자이면서 동시에 다른 가족의 보호자가 될 수 있다.
- 건강정보 공유는 초대, 수락, 공유 권한 토글, 감사 로그를 전제로 한다.
- 본인 동의 없이 건강정보에 접근할 수 없다.
- 식단 사진, 검진표 원본, 복약 정보는 민감정보로 취급한다.
- 가족 연결, 연결 해제, 권한 변경, 건강정보 조회는 감사 로그 대상으로 본다.

이 문서는 `REQ-FAMILY-001`부터 `REQ-FAMILY-011`, `REQ-USER-009`, `REQ-SETTING-008`, `REQ-NOTI-010`, `NFR-PRIV-006`, `NFR-SEC-010`, `NFR-LOG-006`을 구현하기 위한 설계 기준이다.

## 2. 사용자 시나리오

### 2.1 가족 관리 페이지 진입

대상 요구사항: `REQ-FAMILY-001`

1. 사용자가 로그인 후 마이페이지 또는 사이드바의 가족 관리 메뉴로 진입한다.
2. 서버는 사용자가 속한 가족 그룹 목록과 각 그룹의 가족 구성원 목록을 반환한다.
3. 사용자는 가족 목록, 초대 상태, 공유 권한 상태, 최근 알림을 확인한다.

### 2.2 가족 목록 조회

대상 요구사항: `REQ-FAMILY-002`

1. 사용자는 본인이 속한 `families` 목록을 조회한다.
2. 각 가족 그룹에는 `family_members`가 포함된다.
3. 구성원별 표시 정보는 이름, 관계, 가족 내 역할, 가입 여부, 공유 권한 요약, 최근 건강 이벤트 정도로 제한한다.
4. 민감 수치 상세는 `family_share_settings`가 허용한 경우에만 조회한다.

### 2.3 미가입 가족 직접 등록

대상 요구사항: `REQ-FAMILY-003`

1. 사용자가 이름, 관계, 생년월일 또는 메모 수준의 최소 정보를 입력해 미가입 가족을 등록한다.
2. 서버는 `family_members`에 `user_id = null`, `display_name`, `relationship`, `member_status = UNREGISTERED`로 저장한다.
3. 미가입 가족은 앱 계정이 없으므로 실제 로그인 사용자와 연결되지 않는다.
4. 미가입 가족의 건강정보는 본인 동의 전에는 실제 개인 건강 데이터가 아니라 보호자가 입력한 관리용 메모/기록으로만 취급한다.

### 2.4 초대 코드 생성

대상 요구사항: `REQ-FAMILY-004`, `NFR-SEC-010`

1. 가족 관리자가 가족 구성원을 초대하기 위해 초대 코드를 생성한다.
2. 서버는 `family_invites`에 해시된 초대 코드, 만료 시간, 초대 대상 관계, 초대 권한 범위, 초대자를 저장한다.
3. 원본 초대 코드는 생성 직후 한 번만 응답한다.
4. 초대 코드는 만료 시간, 사용 횟수 제한, 재발급, 폐기 상태를 가진다.

### 2.5 초대 코드로 가족 연결

대상 요구사항: `REQ-FAMILY-005`

1. 초대받은 사용자가 앱에서 초대 코드를 입력한다.
2. 서버는 초대 코드 해시, 만료 여부, 사용 가능 여부를 검증한다.
3. 유효하면 `family_members`에 해당 사용자를 `PENDING` 또는 바로 `ACTIVE` 상태로 연결한다.
4. 초대 정책에 따라 초대자 승인 또는 초대받은 사용자 수락이 필요할 수 있다.

### 2.6 가입 사용자 검색 초대

대상 요구사항: `REQ-FAMILY-006`

1. 사용자가 이메일 또는 휴대폰 번호로 가입 사용자를 검색한다.
2. 서버는 개인정보 노출을 줄이기 위해 마스킹된 이름/이메일/휴대폰만 반환한다.
3. 초대자는 검색된 사용자에게 가족 초대를 보낸다.
4. 초대받은 사용자는 알림 또는 초대함에서 수락/거절할 수 있다.

### 2.7 초대 수락/거절

대상 요구사항: `REQ-FAMILY-007`

1. 초대받은 사용자는 초대 목록을 조회한다.
2. 초대 상세에서 가족 그룹명, 초대자, 요청 관계, 공유 요청 범위를 확인한다.
3. 사용자가 수락하면 `family_members.status = ACTIVE`가 된다.
4. 사용자가 거절하면 `family_invites.status = DECLINED`가 되고 가족 구성원 연결은 생성하지 않거나 `DECLINED` 상태로 남긴다.
5. 수락/거절 이력은 audit log에 남긴다.

### 2.8 가족 연결 해제

대상 요구사항: `REQ-FAMILY-008`, `NFR-LOG-006`

1. 사용자는 가족 연결을 해제할 수 있다.
2. 연결 해제 시 `family_members.status = REMOVED` 또는 `ended_at`을 기록한다.
3. 연결 해제 후 건강정보 공유 권한은 즉시 비활성화한다.
4. 연결 해제 이력은 `family_activity_logs` 또는 공통 `audit_logs`에 기록한다.

### 2.9 미가입 가족 앱 가입 시 자동 전환

대상 요구사항: `REQ-FAMILY-009`

1. 미가입 가족이 초대 링크 또는 동일 휴대폰/이메일로 앱에 가입한다.
2. 서버는 가입 정보와 `family_invites` 또는 `family_members.pending_identifier`를 비교한다.
3. 일치하고 사용자 동의가 있으면 `family_members.user_id`를 신규 사용자 id로 연결하고 `member_status = ACTIVE`로 전환한다.
4. 기존 미가입 기록은 신규 사용자 소유 데이터로 자동 이전하지 않는다. 데이터 이전은 별도 동의와 이관 정책이 필요하다.

### 2.10 보호자 알림

대상 요구사항: `REQ-FAMILY-010`, `REQ-FAMILY-011`, `REQ-NOTI-010`

1. 가족 구성원의 건강분석 결과가 생성되면 공유 권한을 확인한다.
2. `analysis_result_share_enabled`가 허용되어 있으면 보호자에게 요약 알림을 생성한다.
3. 혈압, 혈당, 복약 미수행 등 이상 수치가 감지되면 `abnormal_metric_alert_enabled`를 확인한다.
4. 알림 내용은 민감정보 최소화 원칙에 따라 “확인이 필요한 건강 이벤트가 있습니다” 수준으로 제한하고, 상세는 앱 내부 권한 검증 후 보여준다.

## 3. 권한 모델

### 3.1 `users.role`

`users.role`은 시스템 권한만 담당한다.

예:

- `USER`: 일반 사용자
- `ADMIN`: 운영 관리자

가족 보호자 여부를 `users.role = GUARDIAN`으로 표현하지 않는다. 한 사용자가 여러 가족 그룹에서 서로 다른 역할을 가질 수 있기 때문이다.

### 3.2 `family_members.member_role`

가족 안에서의 역할은 `family_members.member_role`로 관리한다.

권장 값:

- `OWNER`: 가족 그룹 생성자 또는 대표 관리자
- `GUARDIAN`: 보호자 역할. 공유 허용된 건강정보를 확인하고 알림을 받을 수 있다.
- `MEMBER`: 본인 건강정보를 공유하거나 가족 그룹에 참여하는 일반 구성원
- `DEPENDENT`: 보호 대상자. 미성년자 또는 보호가 필요한 가족 구성원에 사용 가능

한 사용자는 A 가족 그룹에서는 `MEMBER`, B 가족 그룹에서는 `GUARDIAN`일 수 있다.

### 3.3 `family_share_settings`

가족 공유 권한은 가족 구성원 단위 또는 공유 대상 쌍 단위로 관리한다.

권장 공유 범위:

- 건강분석 결과 공유
- 건강 수치 요약 공유
- 이상 수치 알림 공유
- 챌린지 진행 상태 공유
- 복약 정보 공유
- 식단 기록 공유
- 검진표 OCR 결과 공유
- 원본 파일 공유

민감도가 높은 정보는 기본값을 `false`로 둔다.

민감정보 기본 정책:

- 식단 사진: 기본 비공유
- 검진표 원본 파일: 기본 비공유
- 복약 상세: 기본 비공유
- 건강분석 요약: 명시 동의 후 공유 가능
- 이상 수치 알림: 명시 동의 후 공유 가능

## 4. DB 설계

아래 테이블은 풀서비스형 MVP 범위의 가족 관리 1차 구현 대상이다. 소셜 로그인과 웨어러블 연동을 제외한 현재 MVP 기준에서는 가족 DB/API도 포함 범위로 본다.

### 4.1 `families`

가족 그룹을 표현한다.

권장 컬럼:

- `id BIGSERIAL PK`
- `name VARCHAR(100) NOT NULL`
- `owner_user_id BIGINT NOT NULL FK users.id ON DELETE CASCADE`
- `description TEXT`
- `is_active BOOLEAN NOT NULL DEFAULT TRUE`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`

권장 인덱스:

- `owner_user_id`
- `is_active`

### 4.2 `family_members`

가족 그룹과 사용자 또는 미가입 가족 구성원을 연결한다.

권장 컬럼:

- `id BIGSERIAL PK`
- `family_id BIGINT NOT NULL FK families.id ON DELETE CASCADE`
- `user_id BIGINT FK users.id ON DELETE SET NULL`
- `display_name VARCHAR(100)`
- `relationship VARCHAR(50)`
- `member_role VARCHAR(30) NOT NULL`
- `member_status VARCHAR(30) NOT NULL DEFAULT 'PENDING'`
- `pending_email VARCHAR(255)`
- `pending_phone_number VARCHAR(30)`
- `joined_at TIMESTAMPTZ`
- `ended_at TIMESTAMPTZ`
- `created_by_user_id BIGINT FK users.id ON DELETE SET NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`

권장 인덱스:

- `family_id`
- `user_id`
- `family_id, user_id`
- `family_id, member_status`
- `pending_email`
- `pending_phone_number`

제약:

- 가입 사용자는 `user_id`를 가진다.
- 미가입 가족은 `user_id = null`이고 `display_name`을 가진다.
- 동일 가족 내 동일 활성 `user_id` 중복 연결을 방지한다.

### 4.3 `family_invites`

초대 코드와 가입 사용자 초대를 관리한다.

권장 컬럼:

- `id BIGSERIAL PK`
- `family_id BIGINT NOT NULL FK families.id ON DELETE CASCADE`
- `inviter_user_id BIGINT NOT NULL FK users.id ON DELETE CASCADE`
- `invitee_user_id BIGINT FK users.id ON DELETE SET NULL`
- `invitee_email VARCHAR(255)`
- `invitee_phone_number VARCHAR(30)`
- `invite_code_hash VARCHAR(255)`
- `invite_type VARCHAR(30) NOT NULL`
- `relationship VARCHAR(50)`
- `member_role VARCHAR(30) NOT NULL DEFAULT 'MEMBER'`
- `status VARCHAR(30) NOT NULL DEFAULT 'PENDING'`
- `expires_at TIMESTAMPTZ NOT NULL`
- `accepted_at TIMESTAMPTZ`
- `declined_at TIMESTAMPTZ`
- `revoked_at TIMESTAMPTZ`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`

권장 인덱스:

- `family_id`
- `inviter_user_id`
- `invitee_user_id`
- `status`
- `expires_at`
- `invite_code_hash UNIQUE`

보안:

- 초대 코드 원문 저장 금지
- 서버에는 해시만 저장
- 만료 시간 필수
- 재시도 제한 또는 rate limit 필요

### 4.4 `family_share_settings`

가족 구성원 간 공유 범위를 관리한다.

권장 컬럼:

- `id BIGSERIAL PK`
- `family_id BIGINT NOT NULL FK families.id ON DELETE CASCADE`
- `owner_user_id BIGINT NOT NULL FK users.id ON DELETE CASCADE`
- `viewer_user_id BIGINT NOT NULL FK users.id ON DELETE CASCADE`
- `analysis_result_share_enabled BOOLEAN NOT NULL DEFAULT FALSE`
- `health_metric_share_enabled BOOLEAN NOT NULL DEFAULT FALSE`
- `abnormal_metric_alert_enabled BOOLEAN NOT NULL DEFAULT FALSE`
- `challenge_share_enabled BOOLEAN NOT NULL DEFAULT FALSE`
- `medication_share_enabled BOOLEAN NOT NULL DEFAULT FALSE`
- `diet_share_enabled BOOLEAN NOT NULL DEFAULT FALSE`
- `exam_ocr_share_enabled BOOLEAN NOT NULL DEFAULT FALSE`
- `original_file_share_enabled BOOLEAN NOT NULL DEFAULT FALSE`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`

권장 인덱스:

- `family_id`
- `owner_user_id`
- `viewer_user_id`
- `owner_user_id, viewer_user_id`

제약:

- `owner_user_id != viewer_user_id`
- 활성 가족 관계가 있을 때만 공유 설정이 유효하다.

### 4.5 `family_activity_logs` 또는 `audit_logs`

가족 연결/해제, 권한 변경, 민감정보 조회 이력을 기록한다.

권장 컬럼:

- `id BIGSERIAL PK`
- `family_id BIGINT FK families.id ON DELETE SET NULL`
- `actor_user_id BIGINT FK users.id ON DELETE SET NULL`
- `target_user_id BIGINT FK users.id ON DELETE SET NULL`
- `action_type VARCHAR(50) NOT NULL`
- `target_type VARCHAR(50)`
- `target_id BIGINT`
- `metadata JSONB`
- `ip_address VARCHAR(45)`
- `user_agent TEXT`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`

권장 인덱스:

- `family_id`
- `actor_user_id`
- `target_user_id`
- `action_type`
- `created_at`

대상 action 예:

- `FAMILY_CREATED`
- `MEMBER_ADDED`
- `INVITE_CREATED`
- `INVITE_ACCEPTED`
- `INVITE_DECLINED`
- `MEMBER_REMOVED`
- `SHARE_SETTING_UPDATED`
- `FAMILY_HEALTH_SUMMARY_VIEWED`
- `SENSITIVE_RECORD_VIEWED`

## 5. API 설계

모든 가족 API는 인증 사용자 기준으로 동작한다. 관리자 API와 혼동하지 않는다.

### 5.1 가족 그룹

- `GET /api/v1/families`
  - 내가 속한 가족 그룹 목록 조회
- `POST /api/v1/families`
  - 가족 그룹 생성
- `GET /api/v1/families/{family_id}`
  - 가족 그룹 상세 조회
- `PATCH /api/v1/families/{family_id}`
  - 가족 그룹명/설명 수정
- `DELETE /api/v1/families/{family_id}`
  - 가족 그룹 비활성화 또는 삭제

권한:

- 조회: 활성 구성원
- 수정/삭제: `OWNER` 또는 권한을 위임받은 `GUARDIAN`

### 5.2 가족 구성원

- `GET /api/v1/families/{family_id}/members`
  - 가족 구성원 목록 조회
- `POST /api/v1/families/{family_id}/members/unregistered`
  - 미가입 가족 직접 등록
- `PATCH /api/v1/families/{family_id}/members/{member_id}`
  - 관계, 표시명, 가족 내 역할 수정
- `DELETE /api/v1/families/{family_id}/members/{member_id}`
  - 가족 연결 해제

권한:

- 미가입 가족 등록: `OWNER`, `GUARDIAN`
- 본인 연결 해제: 본인 가능
- 타인 연결 해제: `OWNER` 또는 권한 있는 `GUARDIAN`

### 5.3 초대

- `POST /api/v1/families/{family_id}/invites/code`
  - 초대 코드 생성
- `POST /api/v1/families/invites/code/accept`
  - 초대 코드 입력 후 연결 요청 또는 수락
- `GET /api/v1/families/invites/received`
  - 내가 받은 초대 목록
- `POST /api/v1/families/invites/{invite_id}/accept`
  - 초대 수락
- `POST /api/v1/families/invites/{invite_id}/decline`
  - 초대 거절
- `POST /api/v1/families/{family_id}/invites/search-user`
  - 가입 사용자 검색
- `POST /api/v1/families/{family_id}/invites/user`
  - 가입 사용자에게 초대 발송
- `POST /api/v1/families/invites/{invite_id}/revoke`
  - 초대 취소

보안:

- 초대 코드 검증 실패 응답은 상세 이유를 과도하게 노출하지 않는다.
- 검색 API는 rate limit 대상이다.
- 가입 사용자 검색 결과는 마스킹한다.

### 5.4 공유 권한

- `GET /api/v1/families/{family_id}/share-settings`
  - 가족 공유 설정 목록 조회
- `GET /api/v1/families/{family_id}/share-settings/me`
  - 내 공유 설정 조회
- `PATCH /api/v1/families/{family_id}/share-settings/{setting_id}`
  - 공유 권한 수정
- `POST /api/v1/families/{family_id}/share-settings`
  - 특정 가족 구성원에게 공유 권한 생성

권한:

- 본인의 건강정보 공유 여부는 본인이 결정한다.
- 보호자는 본인이 볼 수 있는 범위를 수정할 수 없다.
- `OWNER`라도 타인의 건강정보를 동의 없이 열람할 수 없다.

### 5.5 가족 건강 요약과 알림

- `GET /api/v1/families/{family_id}/health-summary`
  - 공유 허용된 가족 건강 요약 조회
- `GET /api/v1/families/{family_id}/members/{member_id}/analysis-summary`
  - 특정 가족 구성원의 공유 허용 분석 요약 조회
- `GET /api/v1/families/{family_id}/alerts`
  - 가족 건강 알림 목록 조회

응답 제한:

- 공유 권한이 없으면 수치 상세 대신 권한 없음 상태를 반환한다.
- 원본 파일 URL은 별도 권한 없이는 반환하지 않는다.

## 6. 프론트 화면 설계

### 6.1 가족 관리 메인

경로 예:

- `/family`

구성:

- 가족 그룹 목록
- 가족 구성원 카드
- 내 가족 내 역할 표시
- 공유 권한 상태 요약
- 받은 초대/보낸 초대 상태
- 가족 건강 알림 요약

### 6.2 미가입 가족 등록 화면

구성:

- 이름
- 관계
- 생년월일 또는 나이
- 메모
- 초대 링크 생성 CTA

주의:

- 미가입 가족 정보는 실제 계정 건강정보가 아님을 안내한다.
- 가입 전에는 민감정보 공유 대상이 될 수 없음을 안내한다.

### 6.3 가입 사용자 초대 화면

구성:

- 이메일/휴대폰 검색
- 마스킹된 검색 결과
- 관계 선택
- 요청할 공유 권한 선택
- 초대 보내기

### 6.4 초대 수락/거절 화면

구성:

- 초대자
- 가족 그룹명
- 요청 관계
- 요청 공유 범위
- 수락
- 거절

### 6.5 가족 공유 설정 화면

경로 예:

- `/settings/family-share`

구성:

- 가족 구성원별 공유 토글
- 분석 결과 공유
- 건강 수치 공유
- 이상 수치 알림
- 복약 정보 공유
- 식단 기록 공유
- 검진표 OCR 결과 공유
- 원본 파일 공유

### 6.6 마이페이지 가족 요약

대상 요구사항: `REQ-USER-009`

마이페이지에는 아래 요약만 표시한다.

- 연결된 가족 수
- 받은 초대 수
- 최근 가족 알림
- 가족 관리 페이지 이동 버튼

상세 건강정보는 마이페이지 요약에서 직접 노출하지 않는다.

## 7. 알림 연동 설계

대상 요구사항: `REQ-FAMILY-010`, `REQ-FAMILY-011`, `REQ-NOTI-010`

알림 발생 조건:

- 가족 초대 수신
- 가족 초대 수락/거절
- 가족 연결 해제
- 공유 권한 변경
- 공유 허용된 건강분석 결과 생성
- 공유 허용된 이상 수치 감지
- 공유 허용된 복약 미수행 이벤트

알림 타입 예:

- `FAMILY_INVITE_RECEIVED`
- `FAMILY_INVITE_ACCEPTED`
- `FAMILY_INVITE_DECLINED`
- `FAMILY_MEMBER_REMOVED`
- `FAMILY_SHARE_SETTING_UPDATED`
- `FAMILY_ANALYSIS_RESULT_CREATED`
- `FAMILY_ABNORMAL_METRIC_DETECTED`
- `FAMILY_MEDICATION_MISSED`

알림 정책:

- 알림 목록에는 민감 수치를 직접 노출하지 않는다.
- 상세 클릭 시 API에서 공유 권한을 다시 확인한다.
- 외부 Push/SMS/Email 발송은 별도 worker와 발송 로그가 필요하다.

## 8. 보안/개인정보 정책

### 8.1 공유 범위 제한

대상 요구사항: `NFR-PRIV-006`

- 가족 연결만으로 모든 건강정보를 공유하지 않는다.
- 공유 범위는 항목별로 명시적으로 켜야 한다.
- 민감도가 높은 원본 파일, 식단 사진, 복약 상세는 기본 비공유다.
- 공유 권한은 언제든 철회할 수 있어야 한다.

### 8.2 초대 코드 보안

대상 요구사항: `NFR-SEC-010`

- 초대 코드 원문 저장 금지
- 초대 코드 해시 저장
- 만료 시간 필수
- 사용 후 폐기
- 재시도 제한
- 검색/초대 API rate limiting
- 초대 코드 조회/검증 audit log 기록

### 8.3 접근 로그

대상 요구사항: `NFR-LOG-006`

아래 이벤트는 로그에 남긴다.

- 가족 생성
- 가족 구성원 추가
- 초대 생성/수락/거절/취소
- 가족 연결 해제
- 공유 권한 변경
- 가족 건강 요약 조회
- 민감정보 상세 조회
- 원본 파일 접근

로그에는 actor, target, action, timestamp, IP, user agent, metadata를 남긴다.

### 8.4 의료 책임 고지

가족 알림은 진단 또는 응급 판정이 아니다.

고위험 수치 또는 이상 수치 안내에는 다음 취지의 문구가 필요하다.

- 본 서비스는 의료 진단/처방을 대체하지 않는다.
- 증상이 있거나 수치가 높게 반복되면 의료진과 상담해야 한다.
- 응급 증상이 있으면 즉시 응급 의료기관을 이용해야 한다.

## 9. 구현 단계

### P0: 가족 연결과 공유 권한의 최소 폐쇄 루프

- `families`, `family_members`, `family_invites`, `family_share_settings`, `family_activity_logs` 모델/migration
- 가족 관리 DTO/Repository/Service
- 가족 목록 조회
- 미가입 가족 등록
- 초대 코드 생성
- 초대 코드 수락
- 초대 수락/거절
- 가족 연결 해제
- 공유 권한 조회/수정
- 감사 로그 기록
- 마이페이지 가족 요약

### P1: 가족 건강 요약과 알림

- 가족 건강 요약 API
- 공유 허용된 분석 결과 요약 조회
- 이상 수치 감지 알림
- 가족 알림 타입 추가
- 알림 목록/읽음 처리와 연결
- 가족 공유 설정 화면

### P2: 풀서비스 고도화

- 가입 사용자 검색 초대
- 미가입 가족 앱 가입 시 자동 전환
- 원본 파일 공유 권한 분리
- 복약 미수행 가족 알림
- 식단 기록 공유
- 외부 Push/SMS/Email 알림
- 관리자 가족 이슈 확인 화면
- 가족 관련 abuse/rate limit 모니터링

## 10. 기존 코드 영향 범위

### Backend

추가 예상 파일:

- `app/models/families.py`
- `app/dtos/families.py`
- `app/repositories/family_repository.py`
- `app/services/families.py`
- `app/apis/v1/family_routers.py`
- 신규 migration

영향 예상:

- `app/apis/v1/__init__.py`: family router 등록
- `app/core/db/databases.py`: 모델 모듈 등록
- `app/models/notifications.py`: 가족 알림 타입은 문자열이면 schema 변경 없이 처리 가능
- `app/services/notifications.py`: 가족 알림 생성 orchestration 추가 가능
- `app/apis/v1/mypage_routers.py`: 가족 요약 추가 가능
- `app/apis/v1/setting_routers.py`: 가족 공유 설정 연결 가능

### Frontend

추가 예상 파일:

- `frontend/src/api/families.ts`
- `frontend/src/pages/FamilyPage.tsx`
- `frontend/src/pages/FamilyInvitePage.tsx`
- `frontend/src/pages/FamilyShareSettingsPage.tsx`
- `frontend/src/components/FamilyMemberCard.tsx`

영향 예상:

- `frontend/src/App.tsx`: route 추가
- `frontend/src/components/Sidebar.tsx`: 가족 관리 메뉴 추가
- `frontend/src/pages/MyPage.tsx`: 가족 요약 추가
- `frontend/src/pages/SettingsPage.tsx`: 가족 공유 설정 진입 추가
- `frontend/src/pages/NotificationPage.tsx`: 가족 알림 표시명 매핑 추가

### AI/Worker

P0에서는 AI Worker 영향 없음.

P1 이후 이상 수치 감지/알림을 worker로 분리할 경우:

- health record 생성/분석 완료 이벤트 발행
- 가족 공유 권한 확인
- 가족 알림 생성
- 외부 발송 worker 연동

## 11. 추후 고려 사항

- 미성년자 계정과 보호자 동의 정책
- 법정대리인 인증 여부
- 가족 간 데이터 이관 정책
- 미가입 가족이 가입했을 때 기존 보호자 입력 데이터의 소유권 처리
- 가족 그룹 내 여러 보호자 간 권한 충돌
- 초대 코드 탈취 대응
- 건강정보 공유 철회 후 캐시/알림 상세 접근 차단
- 가족 공유 데이터 export/delete 정책
- 원본 파일 접근 URL 만료 시간
- 외부 알림 발송 실패 재시도
- 감사 로그 보관 기간
- 관리자 화면에서 가족 분쟁/신고 처리 범위

## 12. 2026-05-23 1차 백엔드 구현 상태

가족 관리 기능의 1차 DB/API 골격을 구현했습니다.

### 구현된 테이블

- `families`
- `family_members`
- `family_invites`
- `family_share_settings`

`family_invites`는 초대 코드 원문을 저장하지 않고 `code_hash`만 저장합니다. 초대 코드 원문은 생성 응답에서 1회만 반환합니다. MVP 기본 만료 시간은 30분입니다.

### 구현된 API

- `POST /api/v1/family/groups`
- `GET /api/v1/family/groups`
- `GET /api/v1/family/groups/{family_id}`
- `PATCH /api/v1/family/groups/{family_id}`
- `DELETE /api/v1/family/groups/{family_id}`
- `GET /api/v1/family/groups/{family_id}/members`
- `POST /api/v1/family/groups/{family_id}/members/unregistered`
- `DELETE /api/v1/family/members/{member_id}`
- `POST /api/v1/family/groups/{family_id}/invites`
- `GET /api/v1/family/invites/me`
- `POST /api/v1/family/invites/{invite_id}/accept`
- `POST /api/v1/family/invites/{invite_id}/decline`
- `POST /api/v1/family/invites/code/accept`
- `GET /api/v1/family/share-settings`
- `GET /api/v1/family/groups/{family_id}/share-settings`
- `PATCH /api/v1/family/share-settings/{setting_id}`

### 1차 정책

- 가족 그룹 생성자는 `OWNER` 구성원으로 자동 등록됩니다.
- 가족 그룹 삭제는 물리 삭제가 아니라 `families.status = REMOVED` soft remove로 처리합니다.
- 미가입 가족은 `family_members.user_id = null`, `is_registered = false`, `status = PENDING_UNREGISTERED`로 저장합니다.
- 공유 권한은 기본값을 모두 `false`로 둡니다.
- 공유 권한 변경은 정보 소유자(`owner_user_id`) 본인만 가능합니다.
- 가족 연결만으로 건강정보가 자동 공유되지 않습니다.

### 후속 작업

- 미가입 가족이 앱 가입 시 email/phone 기반 자동 연결 hook
- 가족 건강분석 결과 알림
- 이상 수치 알림
- 복약 미수행 알림
- 챌린지 미수행 알림
- 가족 공유 데이터 상세 조회 API와 `sensitive_access_logs` 연동

## 13. 2026-05-23 프론트 1차 연결 상태

기존 안내용 `FamilyPage`를 실제 family API와 연결했습니다.

- `frontend/src/api/family.ts`에 가족 그룹, 구성원, 초대, 공유 권한 API client를 추가했습니다.
- `/family` 진입 시 가족 그룹 목록, 내 초대 목록, 선택 그룹 구성원, 선택 그룹 공유 권한을 조회합니다.
- 가족 그룹 생성/이름 변경/해제, 미가입 가족 등록, 초대 코드 생성, 초대 코드 수락, 초대 수락/거절, 공유 권한 토글을 연결했습니다.
- 사용자 화면에는 가족 연결만으로 건강정보가 자동 공유되지 않으며, 공유 권한을 켠 항목만 공개된다는 안내를 표시합니다.

아직 구현하지 않은 범위:

- 가족 알림 발송
- 가족 건강정보 상세 공유 조회
- 미가입 가족 가입 시 자동 전환
- 가족 공유 데이터 조회 시 `sensitive_access_logs` 세분화 적용
