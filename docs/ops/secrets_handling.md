# Secrets Handling

## 1. 금지 명령 및 금지 행동

시연, 발표, 코드 리뷰, 화면공유 중 아래 출력은 공유하지 않는다.

- `docker compose config` 전체 출력 공유 금지
  - compose가 `.env`, `env_file`, `environment` 값을 합쳐 보여줄 수 있다.
- `.env` 파일 캡처/공유 금지
  - 루트 `.env`
  - `frontend/.env`
  - 개인 로컬 `.env.*`
- 실제 API key 직접 출력 금지
  - `OPENAI_API_KEY`
  - `LANGFUSE_SECRET_KEY`
  - `LANGFUSE_PUBLIC_KEY`
  - `CLOVA_OCR_SECRET_KEY`
  - Twilio/SMTP/DB password 등 외부 서비스 또는 계정 비밀값
- 아래처럼 값을 직접 찍는 명령 금지

```bash
cat .env
printenv
env
docker compose config
docker compose exec fastapi env
docker inspect fastapi
```

필요한 경우에도 전체 출력은 공유하지 말고, 키 이름만 확인하거나 값을 마스킹한다.

## 2. 안전한 확인 명령

서비스 구성과 상태를 확인할 때는 값이 아닌 구조/상태 중심 명령을 우선 사용한다.

```bash
docker compose config --services
docker compose ps
docker compose logs --tail=100 fastapi
curl http://localhost:8000/api/v1/system/health
```

특정 환경변수가 컨테이너에 들어갔는지만 확인해야 한다면 값을 마스킹한다.

```bash
docker compose exec fastapi sh -lc 'python - <<PY
import os

for key in ["DB_HOST", "REDIS_HOST", "REDIS_PORT", "OPENAI_API_KEY", "LANGFUSE_SECRET_KEY", "CLOVA_OCR_SECRET_KEY"]:
    value = os.getenv(key)
    if value:
        print(f"{key}=<set>")
    else:
        print(f"{key}=<unset>")
PY'
```

`.env` 파일에서 키 존재 여부만 보고 싶을 때도 값을 제거한다.

```bash
grep -E "^(OPENAI_API_KEY|LANGFUSE_SECRET_KEY|CLOVA_OCR_SECRET_KEY|DB_HOST|REDIS_HOST)=" .env \
  | sed -E 's/=.*/=<masked>/'
```

Docker compose 설정 일부를 확인해야 할 때는 전체 config 대신 services만 확인한다.

```bash
docker compose config --services
```

## 3. Git 추적 확인 명령

민감키가 Git 추적 파일에 들어갔는지 확인할 때는 아래 명령을 사용한다. 출력에 실제 키가 포함될 수 있으므로 공유 전 반드시 마스킹한다.

```bash
git grep -n "sk-proj\|OPENAI_API_KEY=.*sk-\|LANGFUSE_SECRET_KEY=.*sk-\|CLOVA_OCR_SECRET_KEY=.*" -- . ':!uv.lock' || true
```

Git이 `.env` 계열 파일을 추적 중인지 확인한다.

```bash
git ls-files | grep -E '(^|/)\.env$|\.env\.|envs/.*\.env' || true
```

과거 커밋에 민감키 패턴이 들어갔는지 확인한다.

```bash
git log -S "OPENAI_API_KEY" --oneline -- . ':!uv.lock'
git log -S "LANGFUSE_SECRET_KEY" --oneline -- . ':!uv.lock'
git log -S "CLOVA_OCR_SECRET_KEY" --oneline -- . ':!uv.lock'
git log -S "sk-proj" --oneline -- . ':!uv.lock'
```

명령 결과에 실제 키가 보이면 채팅/문서/이슈에 그대로 붙이지 않는다.

## 4. Docker 환경변수 주의

- compose `environment`와 `env_file` 값은 이미지에 박히는 build-time secret이 아니라 컨테이너 런타임에 주입된다.
- 하지만 런타임 환경변수는 아래 경로에서 보일 수 있다.
  - `docker compose config`
  - `docker inspect`
  - 컨테이너 내부 `env` 또는 `printenv`
  - 애플리케이션이 실수로 남긴 로그
- 따라서 compose 파일에 직접 운영키를 쓰지 말고 `.env` 또는 secret manager를 사용한다.
- `.env`는 Git에 올리지 않는다.
- `envs/example.*.env`에는 더미값, 빈 값, 문서용 placeholder만 둔다.

## 5. 발표/화면공유 전 체크리스트

- 터미널에 `docker compose config`, `env`, `printenv`, `cat .env` 출력이 남아 있지 않은지 확인한다.
- IDE에서 루트 `.env`, `frontend/.env`, 개인 키 파일을 닫는다.
- 발표 중 열 파일은 `README.md`, `docs/`, 코드 파일, OpenAPI 화면 위주로 제한한다.
- 실제 운영키를 화면공유 또는 채팅에 노출했다면 즉시 재발급한다.
- example env에는 더미값만 유지한다.
- Docker Desktop, 터미널 scrollback, 브라우저 관리 콘솔 화면에도 key가 보이지 않는지 확인한다.

## 6. 노출 발견 시 조치

1. 해당 키를 즉시 폐기 또는 rotate한다.
2. 노출된 위치를 확인한다.
   - Git tracked file
   - commit history
   - issue/comment/chat
   - CI log
   - Docker log
3. Git history에 들어간 경우 단순 파일 삭제만으로는 부족하다.
4. 공유된 로그나 문서에는 키 값을 남기지 않고 `<masked>`로 대체한다.
5. 재발급된 키는 개인 `.env` 또는 운영 secret store에만 보관한다.
