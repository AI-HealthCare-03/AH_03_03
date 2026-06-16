# OCR 엔진 비교 프로젝트

## 개요
건강검진표 이미지/PDF에서 4대 만성질환 관련 수치를 추출하는 OCR 엔진을 비교 테스트합니다.
최종 선정된 OCR은 `ai_worker/vision/ocr/`에 통합될 예정입니다.

> Archive note: 이 문서는 초기 OCR 엔진 비교 실험 기록입니다. 서비스 런타임에서 사용하지 않는 `ocr-compare/` 실행 스크립트는 제거했고, Markdown 기록만 `docs/archive/ocr-compare/`에 보존합니다.

## 테스트 기간
2026-05-18 ~ 2026-05-22 (이번 주 금요일)

## 디렉토리 구조
docs/archive/ocr-compare/
├── README.md                    # 이 파일
├── results/
│   ├── README.md                # 전체 비교 결과
│   ├── summary.md               # 최종 비교표
│   ├── TEMPLATE.md              # 각 OCR README 작성 템플릿
│   ├── columns.json             # 추출 대상 컬럼 정의
│   ├── ground_truth.json        # 정답 데이터
│   └── images/
│       ├── checkup/             # 건강검진표 테스트 이미지/PDF
│       ├── diet/                # 식단 (추후)
│       └── medication/          # 복약 직접 입력 정책으로 전환 전 검토 영역
├── 01_tesseract/                # Tesseract.js
├── 02_python_ocr/               # EasyOCR / PaddleOCR
├── 03_naver_clova/              # Naver CLOVA OCR
├── 04_gpt_vision/               # GPT Vision API
├── 05_google_vision/            # Google Cloud Vision
├── 06_aws_textract/             # AWS Textract
└── 07_azure_document/           # Azure AI Document Intelligence

## 비교 기준
- **정확도**: 11개 컬럼 중 정답과 일치한 비율 (%)
- **속도**: 이미지 1장 처리 시간 (ms)
- **비용**: 무료/부분유료/유료 여부 및 단가

## 추출 대상 컬럼 (11개)
| 질환 | 컬럼 |
|------|------|
| 고혈압 | 수축기 혈압, 이완기 혈압 |
| 당뇨 | 공복혈당 |
| 이상지질혈증 | 총콜레스테롤, 중성지방, HDL, LDL |
| 비만 | 키, 체중, BMI, 허리둘레 |

> 컬럼 변경 시 `results/columns.json` 수정

## 테스트 데이터
| 파일 | 형식 | 비고 |
|------|------|------|
| 신재욱 테스트.jpg | JPG | 건강검진표 이미지 |
| 신재욱_건강검진자료.pdf | PDF | 건강검진표 PDF |
| 윤재님 테스트.jpg | JPG | 건강검진표 이미지 |
| 윤재님 건강검진자료.pdf | PDF | 건강검진표 PDF |
| 지연님 테스트.jpg | JPG | 건강검진표 이미지 |
| 지연님 건강검진자료.pdf | PDF | 건강검진표 PDF |

## 진행 현황
| # | 엔진 | 상태 | 담당 | 완료일 |
|---|------|------|------|--------|
| 1 | Tesseract.js | 🔄 진행 중 | 신재욱 | 2026-05-18 |
| 2 | Python OCR (EasyOCR/PaddleOCR) | ⏳ 대기 | 신재욱 | 2026-05-19 |
| 3 | Naver CLOVA OCR | ⏳ 대기 | 신재욱 | 2026-05-20 |
| 4 | GPT Vision API | ⏳ 대기 | 신재욱 | 2026-05-20 |
| 5 | Google Cloud Vision | ⏳ 대기 | 신재욱 | 2026-05-21 |
| 6 | AWS Textract | ⏳ 대기 | 신재욱 | 2026-05-21 |
| 7 | Azure AI Document Intelligence | ⏳ 대기 | 신재욱 | 2026-05-22 |

## 최종 결과
> 테스트 완료 후 `results/summary.md` 참고
