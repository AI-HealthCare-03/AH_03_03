# 프론트 시연 QA 체크리스트

이 문서는 시연 직전 브라우저에서 실제 클릭 순서로 확인하기 위한 체크리스트다. 민감키, 실제 운영키, `.env`, `docker compose config` 전체 출력은 화면 공유하지 않는다.

## 공통 준비

- 실행 상태 확인:

```bash
docker compose ps
curl http://localhost:8000/api/v1/system/health
```

- 실패 시 로그:

```bash
docker compose logs --tail=100 fastapi
```

- 권장 접속:
  - 프론트: `http://localhost:5173`
  - API docs: `http://localhost:8000/docs`
- API 경로 정합성 기준:
  - 프론트 API client는 `/api/v1` prefix를 `frontend/src/api/client.ts`에서 붙인다.
  - 페이지에서 404가 나면 먼저 브라우저 Network 탭의 요청 path와 FastAPI Swagger의 path를 비교한다.

## 1. 로그인

- 화면 경로: `/login`
- API 연결: `POST /api/v1/auth/login`
- 확인 항목:
  - demo 계정으로 로그인 성공
  - 로그인 후 사이드바/상단바가 보임
  - 일반 USER에게 관리자 콘솔 메뉴가 노출되지 않음
- 눌러볼 버튼:
  - 로그인
  - 아이디 찾기
  - 비밀번호 재설정
- 실패 시:
  - `docker compose logs --tail=100 fastapi`
  - seed 계정 생성 여부 확인
- 상태: 완료

## 2. 회원가입/이메일 인증 화면

- 화면 경로: `/signup`
- API 연결:
  - `POST /api/v1/auth/email-verifications/send`
  - `POST /api/v1/auth/email-verifications/verify`
  - `POST /api/v1/auth/signup`
- 확인 항목:
  - 주소 입력 없이 진행 가능
  - 기본정보와 건강정보 단계가 구분됨
  - 로컬 환경에서 debug code 흐름이 화면을 막지 않음
  - 가입 완료 후 OCR 선택 CTA 또는 나중에 입력하기 흐름이 자연스러움
- 눌러볼 버튼:
  - 이메일 인증 요청/확인
  - 건강검진표 OCR로 정밀 분석 정보 추가
  - 나중에 입력하기
- 실패 시:
  - FastAPI auth 로그
  - `EMAIL_ENABLED`, `EMAIL_VERIFICATION_DEBUG`, SMTP 로컬 설정 확인
- 상태: 부분

## 3. 건강정보 입력

- 화면 경로: `/health`, `/health/profile`
- API 연결:
  - `GET /api/v1/health/records/latest`
  - `POST /api/v1/health/records`
  - `PATCH /api/v1/health/records/{record_id}`
  - `GET /api/v1/health/analysis-readiness`
- 확인 항목:
  - 기본 건강정보 저장 가능
  - readiness에서 basic/precision 상태가 표시됨
  - 누락 필드가 한글 라벨로 보임
- 눌러볼 버튼:
  - 저장
  - 최신 기록 불러오기
- 실패 시:
  - `GET /api/v1/health/analysis-readiness` 응답 확인
- 상태: 완료

## 4. 건강검진 OCR 업로드/confirm

- 화면 경로: `/ocr/exam`
- API 연결:
  - `POST /api/v1/exams`
  - `POST /api/v1/exams/{exam_id}/ocr`
  - `PATCH /api/v1/exams/measurements/{measurement_id}`
  - `POST /api/v1/exams/{exam_id}/confirm`
- 확인 항목:
  - 자동 인식 실행 후 측정값 목록 표시
  - 측정값 수정 가능
  - 확인/저장 후 HealthRecord X2 필드에 반영
  - 이후 정밀분석 readiness가 올라감
- 눌러볼 버튼:
  - 검진표 등록
  - OCR 실행
  - 측정값 수정
  - 확인/저장
- 실패 시:
  - ExamMeasurement 저장 여부
  - `docker compose logs --tail=100 fastapi`
- 상태: 완료

## 5. 분석 readiness 확인

- 화면 경로: `/analysis`
- API 연결: `GET /api/v1/health/analysis-readiness`
- 확인 항목:
  - 간편 분석 가능 여부 표시
  - 정밀 분석 가능 여부 표시
  - 부족한 검진값 안내 표시
- 눌러볼 버튼:
  - 건강정보 입력하기
  - 검진표 OCR 추가하기
- 실패 시:
  - health record 최신 데이터와 X2 필드 확인
- 상태: 완료

## 6. 간편분석 실행

- 화면 경로: `/analysis`
- API 연결: `POST /api/v1/analysis/run-async`, `GET /api/v1/jobs/{job_id}`
- 요청 핵심: `mode=BASIC`
- 확인 항목:
  - 버튼 클릭 후 결과 카드 갱신
  - 위험도/점수/주요 요인 카드가 깨지지 않음
- 눌러볼 버튼:
  - 간편 분석 실행
  - 결과 상세 보기
- 실패 시:
  - readiness missing fields 확인
  - FastAPI logs 확인
- 상태: 완료

## 7. 정밀분석 실행

- 화면 경로: `/analysis`
- API 연결: `POST /api/v1/analysis/run-async`, `GET /api/v1/jobs/{job_id}`
- 요청 핵심: `mode=PRECISION`
- 확인 항목:
  - 정밀분석 버튼이 precision_ready일 때 활성화
  - DM/HTN/DL 결과에 CatBoost badge 또는 model 정보 표시
  - OBESITY는 rule-based로 표시되어도 정상
- 눌러볼 버튼:
  - 정밀 분석 실행
  - 결과 상세 보기
- 실패 시:
  - `uv run python scripts/verify_precision_analysis_api.py`
  - FastAPI ML import/로그 확인
- 상태: 완료

## 8. 분석 결과/히스토리

- 화면 경로:
  - `/analysis/history`
  - `/analysis/{id}`
- API 연결:
  - `GET /api/v1/analysis/results`
  - `GET /api/v1/analysis/results/{result_id}/detail`
- 확인 항목:
  - 목록에서 위험도, 점수, 분석 모드 표시
  - 상세에서 입력 요약과 주요 요인 표시
  - 정밀 결과의 model_name/model_version이 화면을 깨뜨리지 않음
- 눌러볼 버튼:
  - 필터 탭
  - 상세보기 row
  - 전체 리스트
- 실패 시:
  - result id 유효성 확인
- 상태: 완료

## 9. 식단 분석 결과 표시

- 화면 경로:
  - `/diets`
  - `/diets/{id}`
  - `/diets/history`
- API 연결:
  - `POST /api/v1/diets/analyze`
  - `GET /api/v1/diets/{diet_record_id}`
  - `GET /api/v1/diets/{diet_record_id}/photo-result`
- 확인 항목:
  - 식단 분석 실행 후 결과 표시
  - `disease_scores`가 DM/HTN/DL/OBE/ANEM 기준으로 표시
  - `food_score_details`가 일부 표시
  - `scoring_source=nutrition_rule_table` 표시
  - 사용자 화면에 `stub`, `dummy`, `mock` 문구가 노출되지 않음
- 눌러볼 버튼:
  - 식단 분석 실행
  - 직접 입력 저장
  - 기록 완료
  - 추적 대시보드 이동
- 실패 시:
  - DietRecord `nutrition_summary`
  - DietPhotoResult `raw_output`
- 상태: 완료

## 10. 대시보드 반영

- 화면 경로: `/dashboard`
- API 연결:
  - `GET /api/v1/dashboard/summary`
  - `GET /api/v1/dashboard/health`
  - `GET /api/v1/dashboard/trends`
  - `GET /api/v1/dashboard/diets`
  - `GET /api/v1/dashboard/challenges`
  - `GET /api/v1/dashboard/medications`
- 확인 항목:
  - 분석 전 empty state가 결과처럼 보이지 않음
  - 분석 후 위험도/그래프/식단/챌린지 카드가 표시됨
  - 라이트/다크모드에서 차트와 텍스트가 깨지지 않음
- 눌러볼 버튼:
  - 기간 필터
  - 분석 항목 선택 메뉴
  - 식단 업로드하기
- 실패 시:
  - dashboard API 개별 응답 확인
- 상태: 부분

## 11. 챌린지

- 화면 경로:
  - `/challenges`
  - `/challenges/{id}`
- API 연결:
  - `GET /api/v1/challenges`
  - `GET /api/v1/challenges/my`
  - `POST /api/v1/challenges/{challenge_id}/join`
  - `POST /api/v1/challenges/my/{user_challenge_id}/complete-today`
  - `PATCH /api/v1/challenges/my/{user_challenge_id}/give-up`
- 확인 항목:
  - 목록 카드 compact 표시
  - 참여하기/오늘 수행/포기하기 동작
  - 달력과 오늘 완료 상태 표시
- 눌러볼 버튼:
  - 페이지네이션 이전/다음
  - 참여하기
  - 오늘 수행 완료
  - 포기하기
- 실패 시:
  - user_challenge id 매핑 확인
- 상태: 부분

## 12. 알림

- 화면 경로: `/notifications`
- API 연결:
  - `GET /api/v1/notifications`
  - `GET /api/v1/notifications/reminder-schedules`
  - `POST /api/v1/notifications/reminder-schedules`
  - `PATCH /api/v1/notifications/reminder-schedules/{schedule_id}`
  - `DELETE /api/v1/notifications/reminder-schedules/{schedule_id}`
  - `GET /api/v1/notifications/logs`
- 확인 항목:
  - 받은 알림/알림 예약/발송 이력 탭 표시
  - IN_APP 중심 안내 표시
  - 상대시간 표시가 깨지지 않음
- 눌러볼 버튼:
  - 탭 전환
  - 알림 읽음 처리
  - 예약 생성/수정/비활성화
- 실패 시:
  - notification_logs, reminder_schedules seed 여부 확인
- 상태: 부분

## 13. 관리자 콘솔

- 화면 경로: `/admin`
- API 연결:
  - `GET /api/v1/admin/summary`
  - `GET /api/v1/admin/users/summary`
  - `GET /api/v1/admin/system/health`
- 확인 항목:
  - 관리자 계정으로 접속 가능
  - 일반 USER는 접근 불가
  - MONITOR/OPERATOR/ADMIN/SUPER_ADMIN별 메뉴 노출 정책 확인
- 눌러볼 버튼:
  - 관리자 대시보드
  - 모니터링
  - 로그
  - FAQ 관리
  - 문의 관리
- 실패 시:
  - demo admin seed 여부 확인
- 상태: 완료

## 14. FAQ/문의 관리

- 사용자 화면:
  - `/faqs`
  - `/inquiries`
- 관리자 화면:
  - `/admin/faqs`
  - `/admin/inquiries`
- API 연결:
  - `GET /api/v1/faqs`
  - `POST /api/v1/inquiries`
  - `GET /api/v1/inquiries/my`
  - `GET /api/v1/admin/faqs`
  - `POST /api/v1/admin/faqs`
  - `PATCH /api/v1/admin/faqs/{faq_id}`
  - `DELETE /api/v1/admin/faqs/{faq_id}`
  - `GET /api/v1/admin/inquiries`
  - `POST /api/v1/admin/inquiries/{inquiry_id}/answer`
- 확인 항목:
  - 사용자 FAQ/문의 화면이 기존대로 동작
  - 관리자 FAQ 생성/수정/비활성화 동작
  - 문의 답변 작성 가능
  - 의료 진단 단정 표현 주의 안내가 보임
- 눌러볼 버튼:
  - 사용자 문의 등록
  - 관리자 FAQ 생성/수정/비활성화
  - 문의 상세 보기
  - 답변 저장
- 실패 시:
  - 관리자 role 확인
- 상태: 완료

## 15. 404/ErrorBoundary

- 화면 경로:
  - 존재하지 않는 경로 예: `/abc-test`
  - 렌더링 오류 발생 화면
- 연결:
  - React router `*` route
  - 전역 ErrorBoundary
- 확인 항목:
  - 404 페이지 표시
  - 홈 이동/이전 페이지 이동 가능
  - ErrorBoundary fallback에서 stack trace가 노출되지 않음
- 눌러볼 버튼:
  - 홈으로 이동
  - 이전 페이지로 돌아가기
- 실패 시:
  - `frontend/src/App.tsx`
  - `frontend/src/components/ErrorBoundary.tsx`
- 상태: 완료

## 마지막 점검

```bash
uv run ruff check app scripts ai_runtime tests
uv run ruff format app scripts ai_runtime tests --check
uv run pytest tests
uv run python -c "from app.main import app; print(app.title); print(len(app.openapi().get('paths', {})))"
cd frontend
npm run build
```
