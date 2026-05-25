# Demo Ready Checklist

시연 시작 10분 전에 이 문서 순서대로 확인한다. 실제 API key, `.env`, `docker compose config` 전체 출력은 화면 공유에 노출하지 않는다.

## 1. 빠른 로컬 점검

```bash
git status --short
uv run ruff check app scripts ai_worker tests
uv run ruff format app scripts ai_worker tests --check
uv run pytest tests
uv run python -c "from app.main import app; print(app.title); print(len(app.openapi().get('paths', {})))"
uv run python scripts/verify_demo_ready.py
```

기대 결과:

- ruff/format/pytest 통과
- FastAPI import 성공
- OpenAPI path count 출력
- CatBoost import 성공
- DM/HTN/DL CatBoost artifact 확인
- 식단 nutrition score CSV/rule 로드 확인
- 외부 provider key 누락은 `WARN`으로만 표시될 수 있음

## 2. Docker compose 실행

```bash
docker compose down
docker compose build fastapi
docker compose up -d postgres redis fastapi
docker compose ps
curl http://localhost:8000/api/v1/system/health
```

기대 결과:

- `postgres`, `redis`, `fastapi`가 running 또는 healthy
- `/api/v1/system/health` 응답에서 Redis/DB가 정상

실패 시 확인:

```bash
docker compose logs --tail=100 fastapi
docker compose logs --tail=100 postgres
docker compose logs --tail=100 redis
```

## 3. DB migration 및 seed

기존 DB volume은 삭제하지 않는다. `docker compose down -v`는 시연 직전에 사용하지 않는다.

```bash
docker compose exec fastapi uv run --no-sync aerich upgrade
docker compose exec fastapi uv run --no-sync python scripts/seed_mvp_challenges.py
docker compose exec fastapi uv run --no-sync python scripts/seed_mvp_faqs.py
docker compose exec fastapi uv run --no-sync python scripts/seed_demo_users.py
docker compose exec fastapi uv run --no-sync python scripts/seed_current_user_dashboard_demo.py
```

기대 결과:

- demo/admin/monitor 계정 로그인 가능
- 챌린지/FAQ/대시보드 시연 데이터 존재

## 4. 로그인 확인

화면:

- `http://localhost:5173/login`

확인:

- demo 계정 로그인
- admin 또는 monitor 계정 로그인
- 로그인 실패 시 FastAPI 로그와 브라우저 Network 탭 확인

주의:

- 공개 제출 문서에는 실제 시연 비밀번호를 넣지 않는다.
- 계정 안내가 필요한 경우 내부 발표자용 문서에만 남긴다.

## 5. 건강정보 및 readiness

화면:

- 건강정보 입력 화면
- 분석 화면

확인:

- 건강정보 입력/수정 가능
- 분석 readiness에서 간편 분석 가능 상태 확인
- 건강검진 OCR confirm 이후 정밀 분석 가능 상태 확인

실패 시 확인:

```bash
docker compose logs --tail=100 fastapi
```

## 6. 정밀 분석 API E2E

Docker compose와 seed가 준비된 뒤 실행한다.

```bash
uv run python scripts/verify_precision_analysis_api.py --warmup-ml
```

기대 결과:

- 로그인 성공
- `precision_ready=true`
- `/analysis/run` `PRECISION` 성공
- DM/HTN/DL은 `model_name=catboost`
- OBESITY는 `rule_based`

역할 구분:

- `scripts/verify_demo_ready.py`: import, artifact, nutrition asset, optional health check를 보는 상위 점검
- `scripts/verify_precision_analysis_api.py`: 로그인부터 PRECISION 분석 결과까지 보는 API E2E 점검

## 7. 식단 분석

화면:

- 식단 분석 화면

확인:

- 이미지/텍스트 기반 식단 분석 실행
- `disease_scores`에 `DM`, `HTN`, `DL`, `OBE`, `ANEM` 점수 표시 또는 저장
- `scoring_source=nutrition_rule_table`
- 사용자 화면에 `dummy`, `stub`, `mock` 같은 개발용 표현이 보이지 않음

실패 시 확인:

- 브라우저 Network 탭의 `/api/v1/diets/analyze`
- `docker compose logs --tail=100 fastapi`

## 8. 대시보드

화면:

- 추적 대시보드

확인:

- 분석 결과 요약 카드 표시
- 분석 결과가 없을 때 결과처럼 보이는 fallback 위험도 미노출
- 식단/챌린지/건강 팁 영역이 깨지지 않음

## 9. 관리자 콘솔

화면:

- 관리자 콘솔
- FAQ 관리
- 문의 관리
- 모니터링/로그

확인:

- admin/monitor 계정 role에 따라 접근 가능 메뉴가 다름
- OPERATOR 이상에서 FAQ/문의 관리 가능
- USER 계정에서는 관리자 메뉴 미노출

## 10. 마지막 화면 점검

확인할 화면:

- 로그인
- 회원가입/이메일 인증
- 건강정보
- 건강검진 OCR 업로드/confirm
- 분석 실행/결과/히스토리
- 식단 분석/결과
- 대시보드
- 챌린지
- 알림/리마인더
- 가족 기능
- FAQ/문의
- 관리자 콘솔
- 존재하지 않는 경로 404
- ErrorBoundary fallback

## 11. 시연 중 장애 대응

빠른 확인 명령:

```bash
docker compose ps
curl http://localhost:8000/api/v1/system/health
docker compose logs --tail=100 fastapi
uv run python scripts/verify_demo_ready.py
```

민감정보 주의:

- `.env` 파일을 화면에 띄우지 않는다.
- `docker compose config` 전체 출력은 공유하지 않는다.
- 실제 OpenAI/CLOVA/Langfuse/SMTP 키는 출력하지 않는다.
