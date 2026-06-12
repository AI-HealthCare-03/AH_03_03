# Database Migration Policy

이 문서는 PostgreSQL 기준 Aerich migration 운용 정책을 정리한다.

## 현재 기준

- Aerich 설정 위치: `pyproject.toml`
- Aerich migration scan path: `app/core/db/migrations`
- Tortoise 설정: `app.core.db.databases.TORTOISE_ORM`
- 신규 PostgreSQL baseline migration: `app/core/db/migrations/models/0_20260525004115_init.py`
- legacy migration archive: `experiment/db/aerich_legacy_20260525/models/`

## Clean Baseline 정책

현재 baseline은 `app/models` 기준으로 새 PostgreSQL DB를 처음 초기화하기 위한 migration이다.

신규 로컬 DB 또는 신규 시연 DB는 아래 순서로 준비한다.

```bash
DB_HOST=localhost DB_NAME=<new_database_name> uv run --group app aerich upgrade
DB_HOST=localhost DB_NAME=<new_database_name> uv run python scripts/seed_mvp_challenges.py
DB_HOST=localhost DB_NAME=<new_database_name> uv run python scripts/seed_mvp_faqs.py
DB_HOST=localhost DB_NAME=<new_database_name> uv run python scripts/seed_demo_users.py
DB_HOST=localhost DB_NAME=<new_database_name> uv run python scripts/seed_current_user_dashboard_demo.py --email demo@example.com
```

`seed_current_user_dashboard_demo.py`는 `--email` 인자가 필수다.

## Legacy Migration 보존

기존 `0~18` migration은 삭제하지 않고 Aerich scan path 밖으로 이동했다.

```text
experiment/db/aerich_legacy_20260525/models/
```

이 archive는 과거 개발 이력과 문제 원인 추적용이다. Aerich가 읽는 위치가 아니므로 신규 DB 초기화에는 사용하지 않는다.

## 기존 DB 주의사항

이 baseline은 신규 DB 초기화용이다. 이미 데이터가 들어 있는 개발 DB, 공유 DB, 운영 DB에 그대로 적용하면 안 된다.

기존 DB를 새 baseline 정책으로 전환하려면 별도 데이터 마이그레이션 계획이 필요하다.

- 기존 DB/volume 삭제 금지
- 운영/공유 DB에 baseline 직접 적용 금지
- 기존 데이터 보존이 필요한 경우 별도 dump, restore, data migration 절차 필요

## setup_local_mvp_db.py 위치

`scripts/setup_local_mvp_db.py`는 삭제하지 않고 로컬 편의 wrapper로 유지한다.

이 스크립트는 내부적으로 `scripts/init_local_dev_db.py`를 호출하며, `generate_schemas(safe=True)`와 local-only safe ALTER를 실행한다. 따라서 production/shared DB migration의 기준이 아니다.

신규 PostgreSQL DB 기준으로는 Aerich baseline migration만으로 schema 생성이 가능해야 하며, `setup_local_mvp_db.py`는 로컬 테스트 편의와 과거 drift 보정 용도로만 사용한다.

## 시연 전 검증 기준

별도 test DB에서 아래 항목이 통과해야 한다.

```bash
DB_HOST=localhost DB_NAME=ai_health_migration_baseline_test uv run --group app aerich upgrade
DB_HOST=localhost DB_NAME=ai_health_migration_baseline_test uv run python scripts/seed_mvp_challenges.py
DB_HOST=localhost DB_NAME=ai_health_migration_baseline_test uv run python scripts/seed_mvp_faqs.py
DB_HOST=localhost DB_NAME=ai_health_migration_baseline_test uv run python scripts/seed_demo_users.py
DB_HOST=localhost DB_NAME=ai_health_migration_baseline_test uv run python scripts/seed_current_user_dashboard_demo.py --email demo@example.com
```

기존 `ai_health` DB와 Docker volume은 이 검증 과정에서 삭제하지 않는다.

로컬 터미널에서 Docker Compose PostgreSQL에 접근할 때는 `DB_HOST=localhost`를 명시한다. 컨테이너 내부에서 실행할 때는 compose service name인 `postgres`를 사용할 수 있다.
