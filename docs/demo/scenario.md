# 시연 시나리오

이 문서는 시연자가 실제 화면 클릭 순서로 핵심 흐름을 확인하기 위한 가이드다. 계정/비밀번호를 문서에 남기는 경우 공개 제출 전 제거하거나 별도 내부 문서로 분리해야 한다.

## 0. 준비

- Docker compose 기준으로 `postgres`, `redis`, `fastapi`가 실행 중인지 확인한다.
- 필요 시 로컬/시연 seed를 실행한다.
- 민감키가 보이는 `.env` 또는 `docker compose config` 전체 출력 화면은 공유하지 않는다.
- 발표 설명 기준: 현재 `ai_worker`의 로컬 모델 artifact는 DM/HTN/DL CatBoost 3종이다. OBESITY는 rule-based, ANEM은 공식 분석 결과가 아닌 X2/식단 참고 분류이며, 식단 CV/GPT Vision/OCR/LLM은 provider 또는 skeleton 구조를 갖춘 상태에서 기본 시연 경로는 비용 없는 rule/fallback 중심으로 동작한다.
- LLM/RAG 설명 기준: 공식 API에서 현재 직접 호출되는 LLM runtime은 분석/식단 결과 설명 생성(`ai_worker.llm.explanation_service`)과 keyword RAG reference source 첨부다. 메인 챗봇 LLM 라우터, 추천/챌린지 문구 모듈, 기존 RAG generator는 준비/PoC 영역이며 공식 runtime 연결은 후속 작업으로 설명한다.
- 건강검진 OCR 공식 시연 경로에서는 Clova OCR을 호출하지 않는다. 현재 화면은 완성된 OCR provider 결과가 아니라 provider/fallback 기반 측정값 후보를 보여주고, 사용자가 confirm한 값만 `HealthRecord` X2 필드에 반영하는 구조로 설명한다. PaddleOCR/local OCR 1차와 GPT Vision fallback은 후속 provider 후보이며, GPT Vision fallback은 기본 off 상태에서 정책/env가 켜진 경우에만 후보가 된다.
- 비동기 처리 설명 기준: 현재 FastAPI 라우터와 DB I/O는 async 기반이지만 OCR/CV/ML/LLM workflow는 기존 동기 API 흐름을 유지한다. Redis Stream 기반 `async_jobs` skeleton은 `DEMO_ECHO` job만 처리한다. retry/DLQ, heartbeat, 실제 OCR/CV/ML/LLM 비동기화는 운영 확장용 P2로 설명한다. `AnalysisResult.async_job_id`는 향후 실제 분석 job과 `async_jobs` 연동을 위한 reserved field다.
- 인증 시연 기준: Brevo SMTP 이메일 인증은 live 발송 가능 경로로 설명한다. 휴대폰 인증은 MVP/시연 범위에서 보류하며, 회원가입 필수 인증은 이메일 인증만 사용한다. `phone_number`는 DB/프로필 호환성용 선택값으로 유지하고, 운영 전 SMS 인증이 필요하면 별도 provider를 재검토한다.
- `.env`, example env, `ai_worker` 코드 변경 후 이미 떠 있는 Docker 컨테이너에 반영하려면 `docker compose up -d --force-recreate fastapi ai-worker`로 FastAPI/AI Worker를 재생성한다.

안전한 확인 명령:

```bash
docker compose ps
docker compose logs --tail=100 fastapi
curl http://localhost:8000/api/v1/system/health
```

## 1. 사용자 로그인

- 화면 경로: `/login`
- 계정: `demo@example.com` / `Demo1234!`
- 공개 제출 전 계정 정보는 제거 또는 마스킹한다.
- 기대 결과: 로그인 성공 후 홈 또는 대시보드 진입.

## 2. 건강정보/readiness 확인

- 화면 경로: 건강정보 입력 화면 또는 분석 화면
- API: `GET /api/v1/health/analysis-readiness`
- 기대 결과:
  - `basic_ready=true`
  - 검진값이 있으면 `precision_ready=true`
  - 최신 `health_record_id` 확인 가능

## 3. 건강검진 OCR confirm

- 화면 경로: 검진표 OCR 화면
- API:
  - `POST /api/v1/exams/{exam_id}/ocr`
  - `POST /api/v1/exams/{exam_id}/confirm`
- 기대 결과:
  - provider/fallback 기반 측정값 후보가 `ExamMeasurement`에 저장된다.
  - confirm 후 `HealthRecord` X2 필드에 혈압, 혈당, 지질 수치 등이 반영된다.
  - Clova OCR provider는 PoC/deferred 상태이므로 이 경로에서 호출되지 않는다.
  - 발표 시에는 “자동 판독 완료”가 아니라 “후보값 확인 후 반영” 흐름으로 설명한다.

## 4. 정밀분석 실행

- 화면 경로: 분석 실행 화면
- API: `POST /api/v1/analysis/run`
- 요청 핵심:
  - `mode=PRECISION`
  - 최신 `health_record_id`
- 기대 결과: 정밀분석 결과가 생성된다.

## 5. DM/HTN/DL CatBoost 결과 확인

- 확인 위치: 분석 결과 화면, 대시보드, 또는 DB 조회
- 기대 결과:
  - `DIABETES`: `model_name=catboost`
  - `HYPERTENSION`: `model_name=catboost`
  - `DYSLIPIDEMIA`: `model_name=catboost`
  - 각 결과에 `model_version`, `risk_score`가 존재한다.

## 6. OBESITY rule-based 결과 확인

- 확인 위치: 분석 결과 화면, 대시보드, 또는 DB 조회
- 기대 결과:
  - `OBESITY`: `model_name=rule_based`
  - 현재 비만 CatBoost artifact는 없으므로 rule-based가 정상이다.

## 7. 식단 분석 실행

- 화면 경로: 식단 분석 화면
- API: `POST /api/v1/diets/analyze`
- 기대 결과:
  - 음식명 후보 기반 분석 결과가 저장된다.
  - 자체 CV 모델과 GPT Vision fallback은 아직 공식 호출 경로에 붙지 않았다.

## 8. 식단 질병군별 점수 확인

- 확인 항목:
  - `disease_scores`
  - `food_score_details`
  - `scoring_source=nutrition_rule_table`
- 기대 결과:
  - `DM`
  - `HTN`
  - `DL`
  - `OBE`
  - `ANEM`
  5개 점수가 표시 또는 응답에 포함된다.

## 9. 대시보드 반영 확인

- 화면 경로: `/dashboard`
- 기대 결과:
  - 최근 분석 결과 요약
  - 위험도/추이 카드
  - 식단/챌린지/생활관리 요약이 깨지지 않는다.

## 10. 관리자 콘솔 확인

- 화면 경로: `/admin`
- 계정:
  - `admin@example.com` / `Demo1234!`
  - `monitor@example.com` / `Demo1234!`
- 공개 제출 전 계정 정보는 제거 또는 마스킹한다.
- 기대 결과:
  - `SUPER_ADMIN`은 관리자 콘솔 전체 주요 화면 접근 가능.
  - `MONITOR`는 모니터링 중심 접근.
  - FAQ/문의/모니터링/로그 화면이 기본 동작한다.

## 11. 실패 시 확인

```bash
docker compose ps
docker compose logs --tail=100 fastapi
curl http://localhost:8000/api/v1/system/health
```

정밀분석 검증 스크립트:

```bash
uv run python scripts/verify_precision_analysis_api.py
```
