# Schema Normalization Plan

이 문서는 풀서비스 전환 과정에서 DB 컬럼을 바로 삭제하기 전에 영향 범위를 점검하고, 안전한 정규화 순서를 정리하기 위한 설계 메모입니다. 이번 단계에서는 코드, DB, migration을 수정하지 않고 삭제 후보와 보류 대상을 분류한다.

## 1. 전체 결론

- `HealthRecord`의 x1 생활습관 컬럼은 `smoking_status`, `drinking_frequency`, `drinking_amount`, `walking_days_per_week`, `strength_days_per_week`를 표준 필드로 유지하는 방향이 적절하다.
- `is_smoker`, `drinks_alcohol`, `exercise_days_per_week`는 과거 단순 입력값 또는 호환 필드 성격이 강하므로 legacy 제거 후보로 분류한다.
- 다만 프론트의 최신 기록 초기값 fallback, 회원가입 후 health payload, seed 스크립트, DTO 응답에 아직 legacy 필드가 남아 있어 즉시 삭제하면 화면 초기값과 데모 데이터가 흔들릴 수 있다.
- `MedicationRecord.user_id`는 `Medication.user_id`와 중복될 수 있지만, 현재 대시보드/메인 요약/라우터 권한 체크가 직접 `medication_records.user_id`에 의존한다. 당장 제거보다 유지하면서 service layer에서 `medication.user_id == record.user_id` 일치 검증을 강화하는 편이 안전하다.
- `users.last_login`은 코드에서 갱신하지 않고 `last_login_at`만 표준으로 사용 중이다. 모델/ERD/기존 migration에 남아 있는 legacy 컬럼으로, 추후 migration에서 제거 가능성이 높다.

## 2. HealthRecord 표준 필드와 Legacy 필드

| 구분 | 필드 | 현재 사용처 | 판단 |
|---|---|---|---|
| 표준 | `smoking_status` | `health.py` readiness, `analysis.py` snapshot, DTO, 프론트 입력/표시, seed | 유지 |
| 표준 | `drinking_frequency` | `health.py` readiness, `analysis.py` snapshot, DTO, 프론트 입력/표시, seed | 유지 |
| 표준 | `drinking_amount` | `health.py` readiness, `analysis.py` snapshot, DTO, 프론트 입력/표시, seed | 유지 |
| 표준 | `walking_days_per_week` | `health.py` readiness, `analysis.py` snapshot, DTO, 프론트 입력/표시, seed | 유지 |
| 표준 | `strength_days_per_week` | `health.py` readiness, `analysis.py` snapshot, DTO, 프론트 입력/표시, seed | 유지 |
| legacy 후보 | `is_smoker` | DTO, 프론트 fallback/payload, seed | 삭제 후보, 단계적 제거 필요 |
| legacy 후보 | `drinks_alcohol` | DTO, 프론트 fallback/payload, seed | 삭제 후보, 단계적 제거 필요 |
| legacy 후보 | `exercise_days_per_week` | DTO, 프론트 fallback/payload, seed | 삭제 후보, 단계적 제거 필요 |

### 코드 사용처 요약

- 백엔드 모델: `app/models/health.py`에 표준 필드와 legacy 후보가 모두 존재한다.
- 백엔드 DTO: `app/dtos/health.py`의 create/update/response 모두 표준 필드와 legacy 후보를 함께 노출한다.
- readiness: `app/services/health.py`는 표준 필드만 기본 분석 필수값으로 본다.
- 분석 snapshot: `app/services/analysis.py`는 표준 필드만 `input_features`에 포함한다.
- 프론트 입력:
  - `frontend/src/pages/SignupPage.tsx`는 표준 필드와 legacy 후보를 함께 health payload에 넣는다.
  - `frontend/src/pages/HealthRecordPage.tsx`와 `HealthProfilePage.tsx`는 표준 필드가 없을 때 legacy 값을 fallback으로 읽고, 저장 시 legacy 후보도 함께 보낸다.
  - `frontend/src/api/health.ts` 타입에도 legacy 후보가 남아 있다.
- seed:
  - `scripts/seed_demo_users.py`
  - `scripts/seed_current_user_dashboard_demo.py`
  두 스크립트 모두 표준 필드와 legacy 후보를 함께 채운다.
- ERD: `docs/erd/mvp_erd.dbml`에도 표준 필드와 legacy 후보가 함께 남아 있다.

## 3. HealthRecord 정규화 권장안

### 표준화 정책

- 흡연: `smoking_status`를 표준으로 사용한다.
- 음주: `drinking_frequency`, `drinking_amount`를 표준으로 사용한다.
- 운동: `walking_days_per_week`, `strength_days_per_week`를 표준으로 사용한다.
- `is_smoker`, `drinks_alcohol`, `exercise_days_per_week`는 신규 코드에서 쓰지 않는다.

### 단계적 migration 계획

1. 쓰기 중단
   - 프론트 `SignupPage`, `HealthRecordPage`, `HealthProfilePage`에서 legacy 후보를 payload에 넣지 않는다.
   - seed 스크립트에서도 legacy 후보를 더 이상 생성하지 않는다.

2. 읽기 fallback 제거
   - `record?.is_smoker`, `record?.drinks_alcohol`, `record?.exercise_days_per_week` fallback을 제거한다.
   - 오래된 로컬 데이터는 한 번성 backfill로 표준 필드에 반영한다.

3. DTO 응답 정리
   - `HealthRecordCreateRequest`, `HealthRecordUpdateRequest`, `HealthRecordResponse`에서 legacy 후보를 제거한다.
   - 프론트 타입 `frontend/src/api/health.ts`에서도 제거한다.

4. DB migration
   - 기존 데이터를 표준 필드로 backfill한다.
   - 검증 후 `is_smoker`, `drinks_alcohol`, `exercise_days_per_week` 컬럼을 drop한다.

5. ERD 반영
   - `docs/erd/mvp_erd.dbml`에서 legacy 후보를 제거하고, x1 표준 필드만 남긴다.

## 4. MedicationRecord.user FK 유지/제거 판단

### 현재 구조

- `Medication`은 `user_id`를 가진다.
- `MedicationRecord`도 `medication_id`와 `user_id`를 함께 가진다.
- `app/repositories/medication_repository.py`는 `MedicationRecord.filter(user_id=user_id)`로 직접 조회한다.
- `app/services/main.py`, `app/apis/v1/dashboard_routers.py`는 사용자 기준 최근 복약 기록 조회에 `list_medication_records(user_id=...)`를 사용한다.
- `app/apis/v1/medication_routers.py`는 record update 시 `ensure_owner(record.user_id, user)`로 권한을 확인한다.
- seed 스크립트는 `MedicationRecord.get_or_create(medication=medication, user=user, ...)`처럼 두 FK를 모두 채운다.

### 불일치 가능성

- 현재 생성 라우터는 먼저 `medication.user_id` 소유권을 확인한 뒤 `create_medication_record(medication_id, user.id, ...)`를 호출하므로 일반 API 경로에서는 일치한다.
- 그러나 repository나 seed, 향후 worker/admin 코드가 직접 `MedicationRecord.create(medication_id=..., user_id=...)`를 호출하면 `medication.user_id`와 `record.user_id`가 불일치할 수 있다.

### 유지할 경우

- 장점:
  - 사용자 기준 복약 기록 조회가 단순하고 빠르다.
  - `user_id, scheduled_at` index를 활용할 수 있다.
  - 대시보드/알림/리마인더 조회에서 join을 줄일 수 있다.
- 필요 보완:
  - `create_medication_record`에서 medication을 조회해 `medication.user_id == user_id`를 강제한다.
  - record update/delete 시 가능하면 `record.medication.user_id`도 함께 확인한다.
  - DB 제약은 별도 설계가 필요하다.

### 제거할 경우

- 장점:
  - 데이터 정규화 관점에서 중복 FK를 제거할 수 있다.
  - `MedicationRecord`의 소유자는 `Medication`을 통해 일관되게 판단된다.
- 비용:
  - 모든 사용자 기준 조회를 medication join 기반으로 바꿔야 한다.
  - 대시보드, 메인 요약, 알림 worker, 리마인더 스케줄러에서 query 변경이 필요하다.
  - 기존 `user_id, scheduled_at` index 대체 전략이 필요하다.

### 권장 판단

현재 풀서비스에서 복약 리마인드, 알림 worker, 대시보드 시계열 조회가 예정되어 있으므로 `MedicationRecord.user_id`는 당장 제거하지 말고 유지한다. service layer에서 medication 소유자와 medication record 소유자 일치 검증을 적용하고, 실제 제거는 query 성능과 알림 설계를 확정한 뒤 별도 migration으로 검토한다.

### 2026-05-23 service guard 반영 상태

- `MedicationRecord` 생성 시 `medication.user_id`가 현재 사용자와 일치하는지 service layer에서 검증한다.
- `MedicationRecord.user_id`는 request body에서 받지 않고 서버의 current user 기준으로만 저장한다.
- `MedicationRecord` 조회/수정/삭제 시 `record.user_id`와 `record.medication.user_id`가 일치하는지 확인한다.
- 불일치 데이터는 사용자에게 일반적인 접근 불가/찾을 수 없음 응답으로 처리하고, 컬럼은 유지한다.
- 추후 컬럼 제거 여부는 대시보드, 통계, 알림 worker 조회 구조 안정화 후 재검토한다.

## 5. last_login 제거 가능성

### 현재 상태

- `app/models/users.py`에는 `last_login`과 `last_login_at`이 모두 있다.
- `app/repositories/user_repository.py`의 `update_last_login`은 `last_login_at`만 갱신한다.
- `app/services/auth.py`는 로그인 성공 시 `update_last_login(user.id)`를 호출한다.
- 프론트, seed, 라우터에서 `last_login` 직접 사용처는 검색되지 않았다.
- `docs/erd/mvp_erd.dbml`과 문서에는 `last_login`을 legacy, `last_login_at`을 표준으로 명시하고 있다.

### 판단

- `last_login`은 제거 가능성이 높다.
- 단, 기존 DB와 migration history에 남아 있고 관리자 콘솔에서 최근 로그인 표시를 만들 가능성이 있으므로, 관리자 화면 구현 전에 `last_login_at`만 쓰도록 명확히 고정한 뒤 제거 migration을 진행한다.

### 제거 단계

1. 관리자/사용자 응답 DTO에서 `last_login` 미노출 확인.
2. seed와 운영 데이터에 `last_login_at` backfill 필요 여부 확인.
3. `docs/erd/mvp_erd.dbml`에서 `last_login` 제거.
4. 새 migration으로 `users.last_login` drop.

## 6. 코드 영향 범위

### HealthRecord

- `app/models/health.py`
- `app/dtos/health.py`
- `app/services/health.py`
- `app/services/analysis.py`
- `frontend/src/api/health.ts`
- `frontend/src/pages/SignupPage.tsx`
- `frontend/src/pages/HealthRecordPage.tsx`
- `frontend/src/pages/HealthProfilePage.tsx`
- `scripts/seed_demo_users.py`
- `scripts/seed_current_user_dashboard_demo.py`
- `scripts/init_local_dev_db.py`
- `docs/erd/mvp_erd.dbml`

### MedicationRecord

- `app/models/medications.py`
- `app/dtos/medications.py`
- `app/repositories/medication_repository.py`
- `app/services/medications.py`
- `app/apis/v1/medication_routers.py`
- `app/services/main.py`
- `app/apis/v1/dashboard_routers.py`
- `frontend/src/api/medications.ts`
- `frontend/src/pages/MedicationPage.tsx`
- `frontend/src/pages/DashboardPage.tsx`
- `scripts/seed_demo_users.py`
- `scripts/seed_current_user_dashboard_demo.py`

### last_login

- `app/models/users.py`
- `app/repositories/user_repository.py`
- `docs/erd/mvp_erd.dbml`
- 기존 migration 파일

## 7. 안전한 작업 단계

1. 문서 기준 확정
   - 이 문서의 표준 필드와 legacy 후보를 팀 기준으로 확정한다.

2. 신규 쓰기 중단
   - HealthRecord legacy 후보를 프론트 payload와 seed에서 제거한다.
   - MedicationRecord는 유지하되 service layer 일치 검증을 추가한다.

3. 읽기 fallback 제거
   - 오래된 데이터 보정 후 프론트 fallback을 제거한다.

4. DTO 정리
   - legacy 후보를 request/response DTO에서 제거한다.

5. DB migration
   - HealthRecord legacy 컬럼 drop.
   - `users.last_login` drop.
   - `MedicationRecord.user_id`는 유지 여부를 별도 결정한다.

6. ERD/문서 반영
   - migration과 같은 PR에서 ERD를 맞춘다.

## 8. 지금 삭제하면 안 되는 항목

- `is_smoker`
- `drinks_alcohol`
- `exercise_days_per_week`
- `medication_records.user_id`
- `users.last_login`

이 항목들은 삭제 후보 또는 legacy 후보지만, 현재 프론트 fallback, DTO, seed, dashboard/summary 조회 흐름에 아직 영향이 있다. 바로 삭제하면 런타임 오류나 오래된 데이터 표시 누락이 생길 수 있다.

### 2026-05-23 1단계 반영 상태

- 신규 create/update payload에서는 HealthRecord legacy 후보를 더 이상 쓰지 않는다.
- 프론트 `SignupPage`, `HealthRecordPage`, `HealthProfilePage` 저장 payload는 `smoking_status`, `drinking_frequency`, `drinking_amount`, `walking_days_per_week`, `strength_days_per_week` 중심으로 정리했다.
- 백엔드 `HealthRecordCreateRequest`, `HealthRecordUpdateRequest`에서도 legacy 후보를 제거했다.
- seed 스크립트는 표준 필드만 채우도록 정리했다.
- 응답 DTO와 오래된 데이터 읽기 fallback은 다음 단계 호환을 위해 유지한다.

## 9. 다음 migration 후보

### 후보 A: HealthRecord legacy cleanup

- Drop `health_records.is_smoker`
- Drop `health_records.drinks_alcohol`
- Drop `health_records.exercise_days_per_week`

선행 작업:
- 프론트/seed/DTO에서 legacy 쓰기 제거.
- 표준 필드 backfill.
- 프론트 fallback 제거.

### 후보 B: User last_login cleanup

- Drop `users.last_login`

선행 작업:
- 관리자/사용자 화면에서 `last_login_at`만 사용하는지 확인.
- 기존 데이터가 필요하면 `last_login_at = COALESCE(last_login_at, last_login)` backfill.

### 후보 C: MedicationRecord ownership cleanup 또는 consistency guard

- 단기 권장: 컬럼 drop이 아니라 service guard 추가.
- 장기 검토: `medication_records.user_id` 제거 후 medication join 기반 조회로 전환.

## 10. 다음 구현 지시문 초안

```text
현재 브랜치는 feature/kdu-web입니다.

이번 작업은 schema_normalization_plan.md 1단계 구현입니다.
DB 컬럼 삭제/migration은 아직 하지 말고, HealthRecord legacy 후보 필드 신규 쓰기를 중단해주세요.

대상:
- frontend/src/pages/SignupPage.tsx
- frontend/src/pages/HealthRecordPage.tsx
- frontend/src/pages/HealthProfilePage.tsx
- frontend/src/api/health.ts
- scripts/seed_demo_users.py
- scripts/seed_current_user_dashboard_demo.py

요구:
- health payload에서 is_smoker, drinks_alcohol, exercise_days_per_week 제거
- 표준 필드 smoking_status, drinking_frequency, drinking_amount, walking_days_per_week, strength_days_per_week 유지
- 기존 데이터 읽기 fallback은 이번 단계에서는 유지
- DTO/DB/migration은 수정하지 않음
- 빌드/ruff 검증
```

```text
현재 브랜치는 feature/kdu-web입니다.

MedicationRecord.user_id는 당장 제거하지 않고 유지합니다.
대신 MedicationRecord 생성/수정/조회 service에서 medication.user_id와 record.user_id가 불일치하지 않도록 검증을 추가해주세요.
DB/migration은 수정하지 마세요.
```
