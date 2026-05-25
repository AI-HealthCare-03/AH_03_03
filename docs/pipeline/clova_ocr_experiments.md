# CLOVA OCR PoC 실험 정리

## 목적

현재 CLOVA OCR PoC는 건강검진 결과통보서에 대해 OCR 입력 방식별 품질과 속도를 비교하기 위한 실험입니다.

비교 대상은 다음과 같습니다.

- PDF 파일을 CLOVA OCR에 직접 전송한 결과
- 같은 PDF의 전체 페이지를 JPG 이미지로 변환한 뒤 페이지별로 CLOVA OCR에 전송한 결과

이 실험은 운영 라우터와 연결하지 않은 독립 PoC입니다. 건강검진표 기본 OCR과 provider 런타임 코드는 `ai_runtime/ocr`에 두고, 과거 실험 스크립트와 샘플 산출물은 `etc/ai/ocr/providers/clova_ocr` 아래에 archive/reference 용도로 보관합니다.

## 폴더 구조

```text
ai_runtime/ocr/providers/clova_ocr/
├── clova_client.py
├── extractor.py
├── pdf_converter.py
├── settings.py
├── ground_truth/
├── outputs/
├── parsers/
│   ├── __init__.py
│   └── health_exam_result_parser.py
```

```text
etc/ai/ocr/providers/clova_ocr/experiments/
├── debug_raw_json_pages.py
├── evaluate_required_fields.py
├── evaluate_value_accuracy.py
├── export_ground_truth_template.py
├── run_batch_pdf_vs_image_ocr_compare.py
├── run_clova_ocr_to_txt.py
└── run_pdf_vs_image_ocr_compare.py
```

## 환경변수

CLOVA OCR API 정보는 코드에 하드코딩하지 않고 환경변수로 읽습니다.

```bash
CLOVA_OCR_API_URL="..."
CLOVA_OCR_SECRET_KEY="..."
```

프로젝트 루트 `.env`를 사용할 수 있으며, 실제 키는 Git에 올리지 않습니다.
예시값은 `envs/example.local.env`에만 둡니다.

## 입력/출력 관리

로컬 입력 데이터와 결과 파일은 Git 추적 대상에서 제외했습니다.

- `etc/ai/ocr/providers/clova_ocr/data/pdfs/*`
- `etc/ai/ocr/providers/clova_ocr/data/images/*`
- `ai_runtime/ocr/providers/clova_ocr/outputs/*`
- `ai_runtime/ocr/providers/clova_ocr/ground_truth/*`

각 폴더의 `.gitkeep`만 유지합니다.

## 주요 구현

### CLOVA OCR Client

`clova_client.py`는 CLOVA OCR V2 multipart 요청을 담당합니다.

- `jpg`, `jpeg`, `png`, `pdf` 지원
- `requestId`는 `uuid.uuid4()`
- `timestamp`는 milliseconds
- `X-OCR-SECRET` 헤더 사용
- `requests.post(..., timeout=30)` 적용
- API URL과 Secret Key는 환경변수에서만 로드

### PDF 변환

`pdf_converter.py`는 PyMuPDF를 사용해 PDF를 이미지로 변환합니다.

- 첫 페이지 변환: `convert_pdf_first_page_to_image`
- 전체 페이지 변환: `convert_pdf_all_pages_to_images`
- 전체 페이지는 `page_001.jpg`, `page_002.jpg` 형식으로 저장

PyMuPDF가 필요합니다.

```bash
uv pip install pymupdf
```

`pyproject.toml`, `uv.lock`은 PoC 단계에서 수정하지 않았습니다.

### OCR 결과 추출

`extractor.py`는 CLOVA OCR raw JSON에서 텍스트와 field 정보를 추출합니다.

초기에는 `images[0].fields`만 읽을 가능성이 있었는데, PDF direct OCR raw JSON 확인 결과 `images` 배열에 여러 페이지가 들어오는 것으로 확인했습니다.

따라서 현재는 모든 `images` 페이지를 순회합니다.

- `extract_plain_text`
- `extract_fields`
- `calculate_ocr_metrics`
- `get_low_confidence_fields`
- `save_text`
- `save_json`

각 field에는 다음 값을 포함합니다.

- `text`
- `confidence`
- `bounding_box`
- `page_index`
- `page_number`

## 실험 스크립트

### 단일 이미지 OCR

```bash
uv run python etc/ai/ocr/providers/clova_ocr/experiments/run_clova_ocr_to_txt.py
```

기본 이미지에 대해 OCR을 호출하고 다음을 저장합니다.

- `health_exam_ocr.txt`
- `health_exam_ocr_raw.json`
- `health_exam_ocr_metrics.json`

### PDF direct vs 이미지 OCR 단일 비교

```bash
uv run python etc/ai/ocr/providers/clova_ocr/experiments/run_pdf_vs_image_ocr_compare.py
```

기본 PDF 1개에 대해 다음을 수행합니다.

- PDF 직접 OCR
- PDF 전체 페이지 JPG 변환
- 페이지별 이미지 OCR
- 전체 이미지 페이지 텍스트 병합
- PDF direct와 image all pages 비교

### PDF batch 비교

```bash
uv run python etc/ai/ocr/providers/clova_ocr/experiments/run_batch_pdf_vs_image_ocr_compare.py
```

`data/pdfs` 폴더의 모든 PDF를 sorted 순서로 처리합니다.

PDF별 결과는 다음 위치에 저장됩니다.

```text
ai_runtime/ocr/providers/clova_ocr/outputs/{pdf_stem}/
```

대표 산출물은 다음과 같습니다.

- `pdf_direct_ocr.txt`
- `pdf_direct_raw.json`
- `pdf_direct_metrics.json`
- `image_page_001_ocr.txt`
- `image_page_001_raw.json`
- `image_page_001_metrics.json`
- `image_all_pages_ocr.txt`
- `image_all_pages_metrics.json`
- `compare.json`

전체 요약은 다음 파일로 저장됩니다.

- `outputs/batch_compare_summary.json`
- `outputs/batch_compare_summary.csv`

## PDF direct raw 구조 점검

PDF direct OCR 성능이 낮게 나와 raw JSON 구조를 점검했습니다.

```bash
uv run python etc/ai/ocr/providers/clova_ocr/experiments/debug_raw_json_pages.py
```

이 스크립트는 각 `pdf_direct_raw.json`의 `images` 배열 길이와 페이지별 field 수를 출력합니다.

확인 결과 대부분의 PDF direct raw JSON은 `images_count: 3`을 반환했습니다. 따라서 기존처럼 `images[0]`만 평가한 결과는 첫 페이지만 본 잘못된 비교이며, PDF direct 성능 평가는 전체 페이지 기준으로 다시 계산해야 합니다.

## 평가 지표

### OCR speed/confidence

OCR 호출 시간과 CLOVA OCR confidence 기반 지표를 저장합니다.

- `field_count`
- `avg_confidence`
- `min_confidence`
- `max_confidence`
- `low_confidence_count_under_0_8`
- `low_confidence_count_under_0_7`
- `elapsed_seconds`
- `fields_per_second`
- `extracted_text_length`
- `extracted_line_count`
- `page_count`

주의: confidence는 CLOVA OCR이 반환한 인식 신뢰도입니다. 실제 정답 대비 정확도가 아닙니다.

### Required field extraction rate

```bash
uv run python etc/ai/ocr/providers/clova_ocr/experiments/evaluate_required_fields.py
```

OCR 텍스트에 필수 건강검진 항목명이 존재하는지 평가합니다.

예시 필수 항목:

- 성명
- 검진일
- 키
- 몸무게
- 체질량지수
- 허리둘레
- 혈압
- 혈색소
- 공복혈당
- 총콜레스테롤
- 중성지방
- HDL
- LDL
- 혈청크레아티닌
- eGFR
- AST
- ALT
- 감마지티피
- 요단백
- 흉부촬영
- 과거병력
- 약물치료
- 흡연
- 음주
- 신체활동
- 근력운동
- 종합판정
- 의심질환

이 평가는 항목명 존재 여부 기반입니다. 실제 값이 맞는지는 검증하지 않습니다.

### Value-level accuracy

실제 값 단위 정확도 평가는 사람이 작성한 ground truth JSON이 필요합니다.

먼저 템플릿을 생성합니다.

```bash
uv run python etc/ai/ocr/providers/clova_ocr/experiments/export_ground_truth_template.py
```

생성 위치:

```text
ai_runtime/ocr/providers/clova_ocr/ground_truth/{pdf_stem}.json
```

이 파일은 사람이 원본 PDF를 보고 직접 채웁니다. 개인정보와 검진값이 포함될 수 있으므로 Git에 올리지 않습니다.

이후 평가를 실행합니다.

```bash
uv run python etc/ai/ocr/providers/clova_ocr/experiments/evaluate_value_accuracy.py
```

평가 결과:

- `outputs/value_accuracy_summary.json`
- `outputs/value_accuracy_summary.csv`
- `outputs/{pdf_stem}/value_accuracy_detail.json`

비교 지표:

- `field_count`
- `predicted_count`
- `matched_count`
- `missing_prediction_count`
- `wrong_value_count`
- `field_coverage`
- `value_accuracy`
- `precision_like_accuracy`

비교 규칙:

- ground truth 값이 `null`이면 평가 제외
- ground truth 값이 `비해당`이면 평가 대상에 포함
- 숫자값은 기본 exact 비교
- float는 tolerance `0.01` 허용
- 문자열은 `strip` 후 비교
- 리스트는 set 비교

이상지질혈증 관련 수치(`total_cholesterol`, `triglyceride`, `hdl`, `ldl`)는 숫자 결과와 `비해당` 상태를 모두 평가할 수 있습니다.
예를 들어 ground truth와 prediction이 모두 `비해당`이면 matched로 처리하고, ground truth가 `비해당`인데 prediction이 `null`이면 missing prediction, 참고치 숫자(`200.0미만`, `150.0미만`, `60.0이상`, `30.0이상`, `130.0미만`)를 결과값으로 잡으면 wrong value로 처리합니다.

## 파서

`parsers/health_exam_result_parser.py`는 OCR 텍스트에서 주요 검진값을 정규식과 룰 기반으로 추출합니다.

추출 대상:

- `exam_date`
- `height_cm`
- `weight_kg`
- `bmi`
- `waist_cm`
- `systolic_bp`
- `diastolic_bp`
- `hemoglobin`
- `fasting_glucose`
- `total_cholesterol`
- `triglyceride`
- `hdl`
- `ldl`
- `creatinine`
- `egfr`
- `ast`
- `alt`
- `gamma_gtp`
- `urine_protein`
- `chest_xray`
- `suspected_diseases`
- `lifestyle_smoking`
- `lifestyle_drinking`
- `lifestyle_physical_activity`
- `lifestyle_strength_training`

표 형태가 OCR 과정에서 줄 단위로 깨지기 때문에, 파서는 라벨 주변 숫자를 탐색하는 보수적인 방식입니다. mismatch 결과를 보고 룰을 반복 보정해야 합니다.

## 실행 순서

일반적인 실험 순서는 다음과 같습니다.

```bash
uv run python etc/ai/ocr/providers/clova_ocr/experiments/run_batch_pdf_vs_image_ocr_compare.py
uv run python etc/ai/ocr/providers/clova_ocr/experiments/debug_raw_json_pages.py
uv run python etc/ai/ocr/providers/clova_ocr/experiments/evaluate_required_fields.py
uv run python etc/ai/ocr/providers/clova_ocr/experiments/export_ground_truth_template.py
```

그 다음 `ground_truth/*.json`을 사람이 채운 뒤 실행합니다.

```bash
uv run python etc/ai/ocr/providers/clova_ocr/experiments/evaluate_value_accuracy.py
```

## 주의사항

- 이 PoC는 운영 API와 연결하지 않습니다.
- 기본 건강검진표 OCR(`ai_runtime/ocr/checkup`)과 CLOVA OCR PoC(`ai_runtime/ocr/providers/clova_ocr`)는 독립적으로 관리합니다.
- API URL과 Secret Key는 코드에 하드코딩하지 않습니다.
- `.env`, PDF, JPG, OCR outputs, ground truth JSON은 Git에 올리지 않습니다.
- CLOVA confidence는 실제 정답 대비 accuracy가 아닙니다.
- 실제 값 정확도는 ground truth JSON이 작성된 뒤에만 계산할 수 있습니다.
