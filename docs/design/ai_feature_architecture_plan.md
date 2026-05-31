# AI 기능 아키텍처 설계 계획

이 문서는 AI HealthCare 서비스의 AI 기능을 제품 관점에서 묶어 설명하는 설계 방향 문서다. 현재 구현 완료 상태를 과장하지 않고, 구현 완료/부분 구현/구현 예정/정책 검토 필요 항목을 구분한다.

관련 세부 문서는 아래를 기준으로 함께 본다.

- `docs/design/ai_worker_structure.md`: `ai_runtime` 패키지와 Redis Stream worker 구조
- `docs/design/ai_worker_model_scope.md`: 현재 AI 모델/룰/provider 범위
- `docs/design/llm_runtime_scope.md`: LLM/RAG runtime 공식 경로와 PoC 범위
- `docs/design/cv_food_fallback_policy.md`: 식단 CV/GPT Vision fallback 정책
- `docs/design/challenge_recommendation_design.md`: 챌린지 추천 데이터와 정책
- `docs/design/disease_scope_policy.md`: 공식 질환 분석 저장 범위
- `docs/pipeline/health_analysis_pipeline.md`: 건강위험 분석 파이프라인
- `docs/pipeline/rag_runtime.md`: RAG runtime 책임 영역

## 1. 목적

AI 기능은 아래 사용자 경험을 지원하는 방향으로 설계한다.

- LLM 기반 건강 상담/RAG
- CV 기반 식단 이미지 분석
- GPT Vision API fallback
- 간편/정밀 분석 결과 기반 챌린지 추천
- 식단 분석 결과 기반 질환군별 음식 평가
- 정신건강 관련 키워드 감지 및 챌린지/안전 안내 분기

서비스 응답은 의료 진단이 아니라 건강관리 참고 정보다. 위험도, 추천, 식단 평가, 챗봇 답변은 생활습관 개선과 전문기관 상담 권고를 돕는 범위로 제한한다.

## 2. 전체 구성

| 영역 | 역할 | 주요 입력 | 주요 출력 |
| --- | --- | --- | --- |
| LLM/RAG 영역 | 건강 상담, 분석 결과 해석, 추천 사유 설명 | 사용자 질문, 분석 결과, RAG 문서 | 상담 답변, 근거 문서, 추천 설명 |
| CV/식단 분석 영역 | 식단 이미지에서 음식 후보 추정 | 식단 이미지, provider 결과 | 음식명 후보, confidence, review 필요 여부 |
| 질환 위험 분석 영역 | 간편/정밀 분석 기반 질환 위험도 추정 | 건강기록, 검진 수치, 생활습관 | 질환별 risk score, risk level, 주요 요인 |
| 챌린지 추천 영역 | 분석/식단/행동 기록 기반 생활습관 챌린지 추천 | 위험도, 최근 기록, 수행률 | 추천 챌린지, 추천 사유, 난이도 |
| 안전 정책/위기 키워드 처리 영역 | 진단 금지, 민감/위기 표현 분기 | 사용자 질문, 챗봇 입력, 상담 키워드 | 안전 안내, 전문기관 상담 권고, 응답 제한 |

## 3. LLM/RAG 설계

메인 챗봇은 건강 관련 상담, 분석 결과 해석, 챌린지 추천 사유 설명을 담당한다. 사용자가 입력한 건강정보와 분석 결과를 바탕으로 생활습관 관리 관점의 설명을 제공하되, 질병 진단 또는 치료 지시를 내리지 않는다.

RAG는 청킹, 임베딩, 유사도 검색 기반 구조로 확장할 수 있다. 현재 문서 기반 검색이 준비되어 있는 영역과 향후 vector retrieval로 확장할 영역을 구분한다.

RAG 검색 대상 문서는 아래 후보를 우선 chunking 후 embedding하여 검색한다. 답변에는 가능한 경우 사용한 근거 문서 또는 출처를 함께 표시하는 방향으로 설계한다.

- 질환별 건강관리 기준
- 식단 평가 기준
- 챌린지 정책 문서
- 서비스 FAQ
- 공공기관 건강 가이드
- 서비스 내부 안전/개인정보/민감정보 정책 문서

LangChain 또는 LangGraph는 아래 흐름을 제어하는 후보 기술로 둔다.

- 질의 분기
- 검색 대상 선택
- 안전 필터링
- 응답 생성
- 추천 흐름 제어
- 근거 문서와 응답 정합성 확인

LLM 응답 정책:

- 의학적 진단을 내리지 않는다.
- 확정 표현보다 위험요인 안내와 생활습관 관리 중심으로 답한다.
- 고위험 또는 불확실한 상황에서는 전문기관 상담을 권고한다.
- 정신건강 위기 키워드는 일반 챌린지 추천보다 즉시 도움 안내를 우선한다.
- 답변 근거가 부족하면 단정하지 않고 추가 정보 입력 또는 전문가 상담을 안내한다.
- RAG 근거 문서가 있는 경우 문서 제목, source id, 갱신일 같은 추적 가능한 메타데이터를 함께 노출하는 방향을 검토한다.

## 4. CV/식단 이미지 분석 설계

식단 이미지 분석은 1차적으로 AI-Hub 1000개 음식 분류 데이터 기반 CV 모델을 사용하는 방향으로 설계한다. 모델 평가는 AUC보다 음식 분류 태스크에 맞는 지표를 우선한다.

권장 평가 지표:

- Top-1 Accuracy
- Top-5 Accuracy
- F1-score
- confidence calibration

Provider 선택 방향:

- CV 모델 confidence가 충분하면 CV 결과를 사용한다.
- confidence가 기준 이하이면 GPT Vision API를 fallback으로 호출한다.
- Top-k 후보가 서로 비슷해 1순위 음식명을 신뢰하기 어렵다면 사용자 확인 또는 GPT Vision fallback 대상으로 본다.
- AI-Hub 1000개 음식 taxonomy에 포함되지 않는 OOD(out-of-distribution) 음식으로 판단되면 GPT Vision fallback 대상으로 본다.
- GPT Vision은 음식명, 재료, 조리방식 추정에 사용한다.
- 영양성분 수치는 GPT가 생성하지 않고 공공 영양 DB 매칭 기반으로 산출한다.

이 구조는 음식명 추정과 영양성분 계산을 분리하기 위한 것이다. GPT Vision 결과는 영양 계산의 직접 근거가 아니라 DB 매칭을 위한 후보로만 사용한다.

초기 provider decision 예시는 아래와 같다.

| 조건 | 처리 방향 |
| --- | --- |
| Top-1 confidence가 기준 이상이고 Top-1/Top-2 차이가 충분함 | CV 모델 결과를 후보로 사용 |
| Top-1 confidence가 기준 미만 | GPT Vision fallback 또는 사용자 확인 |
| Top-k 후보가 비슷해 불명확함 | GPT Vision fallback 또는 복수 후보 제시 |
| AI-Hub taxonomy 밖 음식으로 추정됨 | GPT Vision fallback |
| GPT Vision도 불확실함 | 임의 영양 계산 금지, 사용자 confirm 요청 |

## 5. 영양성분 DB 매칭 정책

영양성분 계산은 신뢰 가능한 DB source 매칭을 기준으로 한다.

데이터 출처 후보:

- 식품의약품안전처 식품영양성분 DB
- 농식품올바로 미가공 식품 정보
- 식단영양성분 DB
- AI-Hub 음식 분류/영양 정보
- 자체 음식명 매핑 테이블

정책:

- 음식명 표준화가 필요하다.
- DB source 우선순위와 fallback 정책이 필요하다.
- GPT Vision 결과는 DB 매칭용 후보로만 사용한다.
- 사용자가 음식명/분량을 confirm 또는 수정할 수 있는 구조가 필요하다.
- 매칭 실패 시 임의 영양성분을 생성하지 않고 review 상태로 둔다.

권장 매칭 우선순위:

| 우선순위 | Source | 사용 목적 | 실패 시 |
| --- | --- | --- | --- |
| 1 | 자체 음식명 매핑 테이블 | 서비스에서 쓰는 canonical food name, alias, provider 후보명을 표준화 | 공공 DB 직접 검색 |
| 2 | 식품의약품안전처 식품영양성분 DB | 가공식품/일반 음식 영양성분 기준값 | 식단영양성분 DB 검색 |
| 3 | 식단영양성분 DB | 조리식/외식형 음식의 대표 영양성분 보강 | 농식품올바로 검색 |
| 4 | 농식품올바로 미가공 식품 정보 | 원재료/미가공 식품 기준값 | 사용자 confirm 요청 |
| 5 | AI-Hub 음식 분류/영양 정보 | CV taxonomy와 연결되는 보조 metadata | review 상태 유지 |

음식명 정규화/표준화 단계:

1. Provider 후보 음식명 수집
2. 공백, 괄호, 조리 표현, 브랜드/수식어 정리
3. alias 사전과 자체 음식명 매핑 테이블 조회
4. canonical food name 후보 생성
5. source별 nutrition DB lookup
6. 사용자가 음식명/분량을 confirm 또는 수정
7. 확정된 음식명과 영양성분만 최종 저장

예시 흐름:

```text
image
  -> CV model or GPT Vision
  -> food name candidates
  -> canonical food name mapping
  -> nutrition DB lookup
  -> user confirm/edit
  -> diet record and disease food score
```

## 6. 간편 분석/정밀 분석 연계

간편 분석은 기본 정보, BMI, 생활습관, 가족력 등을 기반으로 위험군을 추정한다. 검진 수치가 없어도 사용자가 직접 입력한 기본 건강정보를 활용해 초기 위험관리 방향을 제시하는 용도다.

정밀 분석은 건강검진 수치까지 반영하여 질환별 위험요인과 관리 우선순위를 제공한다. DM/HTN/DL처럼 모델 artifact가 준비된 질환군은 모델 예측을 사용할 수 있고, 그 외 질환군은 rule 기반 또는 정책 기반 보조 판단으로 구분한다.

분석 결과 활용:

- 챌린지 추천 입력
- 대시보드 위험도 추이 입력
- LLM/RAG 결과 해석 입력
- 식단/운동/복약 기록 우선순위 판단

분석 결과와 챌린지 추천 연결:

- 간편 분석 결과는 초기 위험군과 생활습관 개선 방향을 정하는 입력으로 사용한다.
- 정밀 분석 결과는 검진 수치 기반 위험요인과 관리 우선순위를 반영한다.
- 추천 로직은 질환 위험도만 보지 않고, 개선 가능성, 데이터 신뢰도, 최근 기록, 수행률, 난이도/부담도를 함께 본다.
- LLM은 추천 결정 자체를 단독으로 수행하지 않고, 추천 사유와 기대 개선 효과를 자연어로 설명하는 계층으로 둔다.

질환 예측 모델 평가지표는 목적에 따라 구분한다.

- 위험군 누락을 줄이는 목적: Recall
- 정밀도와 재현율 균형: F1-score
- 전체 threshold 비교와 순위 성능: AUC

## 7. 챌린지 추천 로직

챌린지 추천은 추천 결정 로직과 설명 생성 로직을 분리한다. 추천 자체는 룰엔진/ML 기반으로 안정적으로 결정하고, LLM은 추천 사유와 기대 개선 효과를 사용자 친화적으로 설명하는 역할을 맡을 수 있다.

추천 기준:

- 질환 위험도
- 개선 가능성
- 사용자 입력 데이터 신뢰도
- 최근 식단/운동/복약 기록
- 기존 챌린지 수행률
- 과도하지 않은 난이도/부담도
- 사용자 그룹군

추천 종류:

- 간편/정밀 분석 결과 기반 추천
- 식단 분석 결과 기반 추천
- 생활습관 기록 기반 추천
- 챌린지 수행률 기반 난이도 조정

설계 원칙:

- 질환명을 진단처럼 표시하지 않는다.
- 추천 사유는 위험요인과 개선 가능성 중심으로 설명한다.
- 같은 사용자에게 너무 많은 챌린지를 한 번에 권하지 않는다.
- 포기/미달성 이력을 무시하지 않고 다음 난이도 조정에 참고한다.
- LLM 설명은 "이 챌린지를 하면 질병이 치료된다"가 아니라 "위험 관리와 생활습관 개선에 도움이 될 수 있다"는 수준으로 제한한다.

## 8. 질환군별 식단 평가 기준

식단 평가는 질환군별로 중요 영양소와 식사 패턴을 다르게 본다. 아래 기준은 초기 설계 예시이며, 실제 점수화 기준은 영양 DB source와 임상/공공 가이드 검토 후 조정한다.

| 질환군 | 평가 중심 |
| --- | --- |
| 고혈압 위험군 | 나트륨, 포화지방, 칼륨, 총열량 |
| 당뇨 위험군 | 탄수화물, 당류, 식이섬유, 총열량 |
| 이상지질혈증 위험군 | 포화지방, 트랜스지방, 콜레스테롤, 총지방 |
| 비만 위험군 | 총열량, 지방, 당류, 식사량 |
| 공통 건강관리군 | 균형도, 과식 여부, 단백질/채소 부족 여부 |

사용자에게는 점수 자체보다 "어떤 부분을 조정하면 좋은지"를 설명한다. 예를 들어 나트륨이 높은 식단에는 저염 선택, 국물 섭취 줄이기, 채소 보강 같은 행동 제안을 연결한다.

## 9. 정신건강 키워드 처리

표현은 "정신병"이 아니라 "정신건강 관련 키워드"로 사용한다. 정신건강 키워드 처리는 챌린지 추천보다 안전 정책을 우선한다.

분기 기준:

| 입력 표현 범위 | 처리 방향 |
| --- | --- |
| 일반 스트레스/불안/수면 문제 | 정신건강 챌린지 추천, 수면/호흡/기록 등 자기관리 행동 제안 |
| 우울/무기력/번아웃 표현 | 자기관리 챌린지와 함께 전문 상담 권고 |
| 자해/극단 선택/죽고 싶다 등 위기 키워드 | 챌린지 추천보다 즉시 도움 안내, 전문기관 상담 권고, 보호자/주변 사람 연결 안내 우선 |

위기 키워드가 감지되면 LLM이 일반적인 건강관리 답변을 계속 생성하지 않도록 안전 분기를 먼저 적용한다.
또한 위기 키워드에서는 일반 챌린지 추천 화면으로 바로 보내지 않고, 즉시 도움을 받을 수 있는 안내와 전문기관/보호자 연결 안내를 우선한다.

## 10. Fallback 및 안전 정책

Fallback 정책 후보:

- CV confidence threshold 미달
- Top-k 예측 실패
- OOD 음식 판단
- GPT Vision fallback 조건 충족
- OCR/provider 실패
- 모델 artifact 로드 실패
- nutrition DB 매칭 실패

안전 정책:

- LLM 진단 금지
- 정신건강 위기 키워드 안전 분기
- 민감 건강정보 원문을 Redis payload, push 제목/본문, 외부 로그에 직접 넣지 않기
- GPT Vision/LLM 결과는 사용자 confirm 또는 내부 검증 없이 확정 데이터로 저장하지 않기
- OCR/식단 인식 결과는 오류 가능성을 안내하고 사용자가 수정할 수 있게 하기

사용자 confirm 구조:

- OCR 검진 수치 후보 confirm
- 약봉투 OCR 후보 confirm
- 식단 음식명/분량 confirm
- 추천 챌린지 선택 또는 거절

## 11. 평가 지표

| 영역 | 지표 후보 |
| --- | --- |
| CV 음식 분류 | Top-1 Accuracy, Top-5 Accuracy, F1-score, confidence, fallback rate |
| 질병 위험 예측 | Recall, F1-score, AUC |
| LLM/RAG | 검색 정확도, 근거 문서 일치율, 금지 응답률, hallucination 의심률 |
| 서비스 운영 | fallback 호출률, 사용자 수정률, 추천 수락률, 챌린지 완료율 |

평가 지표는 모델 성능뿐 아니라 서비스 신뢰도를 함께 봐야 한다. 특히 식단과 OCR은 provider가 맞았는지보다 사용자가 최종 저장 전에 얼마나 수정했는지가 중요한 운영 지표가 될 수 있다.

## 12. 로그/추적 항목

추적 후보:

- 어떤 모델/provider가 사용됐는지
- GPT Vision fallback 여부
- 사용된 영양 DB 출처
- 음식명 표준화 결과
- 사용자 confirm 여부
- LLM prompt version
- retrieval document id
- 분석 결과와 챌린지 추천 연결 근거
- 챌린지 추천 사유
- LLM 검색 문서와 응답 로그
- 위기 키워드 감지 여부
- safety 분기 결과
- 사용자 confirm/edit 여부
- async job id, job type, status, retry/DLQ 여부

로그 원칙:

- OCR 원문, 건강정보 원값, 이미지/PDF bytes, 약품명 원문 전체 같은 민감 원문은 외부 로그와 Redis payload에 직접 남기지 않는다.
- 디버깅 로그는 provider, content_type, file_ext, text length, candidate count, fallback 여부 같은 메타데이터 중심으로 남긴다.
- Langfuse 또는 유사 관측 도구를 사용할 때도 prompt/response에 민감정보가 들어가는지 별도 정책이 필요하다.

## 13. 구현 상태 구분

| 구분 | 항목 | 상태 | 비고 |
| --- | --- | --- | --- |
| 구현 완료 | Redis Stream 기반 async job infrastructure | 구현 완료 | retry/backoff, DLQ, pending recovery 구조가 준비되어 있다. |
| 구현 완료 | CatBoost 기반 DM/HTN/DL 정밀 분석 경로 | 구현 완료 | 세부 artifact와 feature mapping은 별도 모델 범위 문서를 기준으로 한다. |
| 구현 완료 | `analysis.run`, `exam_ocr.run`, `diet.analyze_image`, `medication_ocr.run` job type | 구현 완료 | 실제 provider 활성화 여부는 env와 handler 정책에 따른다. |
| 구현 완료 | 식단 질환군별 nutrition scorer | 구현 완료 | 현재 점수표/rule 기반을 사용한다. |
| 부분 구현 | LLM/RAG 설명 생성 | 부분 구현 | rule 기반 설명과 keyword RAG/trace 보조 경로가 있으며 vector RAG는 계획 영역이다. |
| 부분 구현 | GPT Vision provider | 부분 구현 | env flag와 fallback 정책에 따라 사용할 수 있으나 비용/안전 정책이 필요하다. |
| 부분 구현 | 챌린지 추천 | 부분 구현 | master data와 분석 결과 기반 추천 구조가 있으며, LLM 설명 고도화는 계획 영역이다. |
| 구현 예정 | AI-Hub 1000개 음식 분류 CV 모델 공식 연결 | 구현 예정 | 모델 학습/평가/배포 기준과 fallback threshold 정의가 필요하다. |
| 구현 예정 | 영양성분 DB source 우선순위와 표준화 파이프라인 | 구현 예정 | 공공 DB별 필드 차이와 음식명 canonical mapping 정책이 필요하다. |
| 구현 예정 | LangChain/LangGraph 기반 질의 분기 orchestration | 구현 예정 | 현재는 설계 후보이며 공식 runtime 채택 여부는 별도 검토한다. |
| 정책 검토 필요 | 정신건강 위기 키워드 안전 분기 | 정책 검토 필요 | 챌린지 추천보다 즉시 도움 안내를 우선하는 문구/운영 기준 필요. |
| 정책 검토 필요 | LLM/RAG 로그와 민감정보 마스킹 | 정책 검토 필요 | 관측 도구 연동 시 prompt/response 보관 범위 확정 필요. |
| 정책 검토 필요 | GPT Vision fallback 비용/호출 제한 | 정책 검토 필요 | 데모/운영 환경별 호출량, 실패 처리, 사용자 안내 기준 필요. |

## 다음 정리 후보

- 식단 CV 모델 평가 기준과 threshold 별도 문서화
- nutrition DB source 우선순위와 음식명 표준화 정책 문서화
- 정신건강 안전 문구와 crisis keyword 분기 정책 문서화
- RAG source registry와 vector RAG 전환 계획 문서화
- 챌린지 추천 룰엔진/ML/LLM 설명 책임 분리 문서화
