"""
03_naver_clova/test_clova.py

Naver CLOVA OCR 기반 건강검진표 수치 추출 테스트
이미지(JPG) + PDF 지원

실행: python test_clova.py
"""

import json
import os
import re
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── 환경변수 로드 (.env는 AH_03_03 루트) ─────────────────────────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env")

CLOVA_API_URL = os.getenv("CLOVA_API_URL", "")
CLOVA_SECRET_KEY = os.getenv("CLOVA_SECRET_KEY", "")

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # ocr-compare/
IMAGES_DIR = BASE_DIR / "results" / "images" / "checkup"
RESULTS_DIR = BASE_DIR / "results"
COLUMNS_FILE = RESULTS_DIR / "columns.json"
GROUND_TRUTH_FILE = RESULTS_DIR / "ground_truth.json"
OCR_NAME = "clova"

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
with open(COLUMNS_FILE, encoding="utf-8") as f:
    columns_data = json.load(f)

with open(GROUND_TRUTH_FILE, encoding="utf-8") as f:
    ground_truth = json.load(f)

NOT_MEASURED_VALUE = columns_data["not_measured_value"]
NOT_MEASURED_KEYWORDS = columns_data["not_measured_keywords"]

FIELD_KEYS = []
NULLABLE_FIELDS = set()
for category in columns_data["columns"].values():
    for field in category["fields"]:
        FIELD_KEYS.append(field["key"])
        if field.get("nullable"):
            NULLABLE_FIELDS.add(field["key"])

# ── 키워드 매핑 ───────────────────────────────────────────────────────────────
FIELD_KEYWORDS = {
    "systolic_bp": ["수축기", "수축기혈압", "최고혈압", "SBP", "mmHg"],
    "diastolic_bp": ["이완기", "이완기혈압", "최저혈압", "DBP"],
    "fasting_glucose": ["공복혈당(mg", "공복혈당", "공복 혈당", "당뇨병", "GLU"],
    "total_cholesterol": ["총콜레스테롤", "총 콜레스테롤", "콜레스테롤(mg", "TC", "T-CHO"],
    "triglyceride": ["중성지방(mg", "중성지방", "TG"],
    "hdl": ["고밀도 콜레스테롤", "고밀도콜레스테롤", "HDL"],
    "ldl": ["저밀도 콜레스테롤", "저밀도콜레스테롤", "LDL"],
    "height_cm": ["키(cm)", "신장(cm)", "신장", "Height"],
    "weight_kg": ["몸무게(kg)", "체중(kg)", "체중", "몸무게"],
    "bmi": ["체질량지수", "체질량지수(kg", "BMI", "비만도"],
    "waist_cm": ["허리둘레(cm)", "허리둘레", "허리 둘레"],
}

# ── 노이즈 패턴 ───────────────────────────────────────────────────────────────
NOISE_PATTERN = re.compile(r"\d+\.?\d*\s*(만만|미만|이하|이상|초과|이내)")
CHECKBOX_PATTERN = re.compile(r"[■□▣▪●○◆◇]")


def clean_line(text: str) -> str:
    text = CHECKBOX_PATTERN.sub("", text)
    text = NOISE_PATTERN.sub("", text)
    return text.strip()


def extract_numbers(text: str) -> list[float]:
    cleaned = clean_line(text).replace(",", "")
    matches = re.findall(r"\d+\.?\d*", cleaned)
    return [float(m) for m in matches if m]


def is_keyword_match(text: str, keywords: list[str]) -> bool:
    upper = text.upper()
    return any(kw.upper() in upper for kw in keywords)


def is_not_measured(text: str) -> bool:
    return any(kw.upper() in text.upper() for kw in NOT_MEASURED_KEYWORDS)


def validate_value(field: str, value: float) -> bool:
    ranges = {
        "systolic_bp": (60, 250),
        "diastolic_bp": (40, 150),
        "fasting_glucose": (40, 600),
        "total_cholesterol": (50, 600),
        "triglyceride": (20, 2000),
        "hdl": (10, 200),
        "ldl": (20, 500),
        "height_cm": (100, 250),
        "weight_kg": (20, 300),
        "bmi": (10, 70),
        "waist_cm": (40, 200),
    }
    if field not in ranges:
        return True
    low, high = ranges[field]
    return low <= value <= high


# ── 파싱 함수 ─────────────────────────────────────────────────────────────────


def parse_blood_pressure(lines: list[str]) -> tuple:
    for line in lines:
        if is_keyword_match(line, ["수축기", "mmHg", "SBP"]):
            nums = [n for n in extract_numbers(line) if 40 <= n <= 250]
            if len(nums) >= 2:
                return nums[0], nums[1]

    for i, line in enumerate(lines):
        if "mmHg" in line or "mHg" in line:
            context = lines[max(0, i - 3) : i + 3]
            combined = " ".join(context)
            nums = [n for n in extract_numbers(combined) if 40 <= n <= 250]
            if len(nums) >= 2:
                return nums[0], nums[1]
            if len(nums) == 1:
                return nums[0], None

    return None, None


def parse_height_weight(lines: list[str]) -> tuple:
    """
    건강검진표 레이아웃:
      키(cm)
      및
      몸무게(kg)
      175.7
      /
      87.6
    키워드 발견 후 최대 6줄 안에서 유효한 숫자 2개를 순서대로 찾는다.
    """
    for i, line in enumerate(lines):
        if is_keyword_match(line, ["키", "신장", "몸무게", "체중"]):
            context = lines[i : i + 7]
            nums = []
            for ctx_line in context:
                for n in extract_numbers(ctx_line):
                    if 100 <= n <= 250:
                        nums.append(("h", n))
                    elif 20 <= n <= 300 and nums and nums[-1][0] == "h":
                        nums.append(("w", n))
            # h, w 쌍 추출
            for j in range(len(nums) - 1):
                if nums[j][0] == "h" and nums[j + 1][0] == "w":
                    return nums[j][1], nums[j + 1][1]
            # 단순하게 범위만으로 재시도
            all_nums = []
            for ctx_line in context:
                all_nums.extend(extract_numbers(ctx_line))
            candidates_h = [n for n in all_nums if 100 <= n <= 250]
            candidates_w = [n for n in all_nums if 20 <= n <= 150]
            if candidates_h and candidates_w:
                return candidates_h[0], candidates_w[0]
    return None, None


def parse_bmi(lines: list[str]):
    """
    BMI 수치를 파싱한다.
    수치가 없고 구간만 있는 경우(동욱님처럼) None을 반환 → 정확도 계산 제외.
    """
    bmi_noise = re.compile(r"\d+\.?\d*\s*[-~]\s*\d+\.?\d*")
    for i, line in enumerate(lines):
        if not is_keyword_match(line, FIELD_KEYWORDS["bmi"]):
            continue
        cleaned = bmi_noise.sub("", line)
        cleaned = re.sub(r"\d+\.?\d*(미만|이상|이하)", "", cleaned)
        nums = [n for n in extract_numbers(cleaned) if 10 <= n <= 50]
        if nums:
            return nums[0]
        if i + 1 < len(lines):
            next_line = bmi_noise.sub("", lines[i + 1])
            next_line = re.sub(r"\d+\.?\d*(미만|이상|이하)", "", next_line)
            next_nums = [n for n in extract_numbers(next_line) if 10 <= n <= 50]
            if next_nums:
                return next_nums[0]
    return None


def parse_waist(lines: list[str]):
    """
    건강검진표 레이아웃:
      허리둘레(cm)
      91.5
    키워드 매칭 후 최대 3줄 안에서 유효한 숫자를 찾는다.
    """
    for i, line in enumerate(lines):
        if not is_keyword_match(line, FIELD_KEYWORDS["waist_cm"]):
            continue
        for ctx_line in lines[i : i + 4]:
            nums = [n for n in extract_numbers(ctx_line) if 40 <= n <= 200]
            if nums:
                return nums[0]
    return None


def parse_fasting_glucose(lines: list[str]):
    """
    건강검진표 레이아웃:
      공복혈당(mg/dL)
      83
    키워드 매칭 후 최대 3줄 안에서 유효한 숫자를 찾는다.
    """
    for i, line in enumerate(lines):
        if not is_keyword_match(line, FIELD_KEYWORDS["fasting_glucose"]):
            continue
        for ctx_line in lines[i : i + 4]:
            nums = [n for n in extract_numbers(ctx_line) if 40 <= n <= 600]
            if nums:
                return nums[0]
    return None


def parse_dyslipidemia_all(lines: list[str]) -> dict:
    """
    건강검진표 레이아웃:
      중성지방(mg/dL)
      총콜레스테롤(mg/dL)
      고밀도 콜레스테롤(mg/dL)
      저밀도 콜레스테롤(mg/dL)
      [값1]
      [값2]
      [값3]
      [값4]

    4개 키워드가 연속으로 나온 뒤 값들이 순서대로 오는 구조.
    키워드 발견 순서와 값 순서를 매핑한다.
    단일 키워드 파싱 fallback도 포함.
    """
    DYSLIPIDEMIA_KEYWORDS = {
        "triglyceride": FIELD_KEYWORDS["triglyceride"],
        "total_cholesterol": FIELD_KEYWORDS["total_cholesterol"],
        "hdl": FIELD_KEYWORDS["hdl"],
        "ldl": FIELD_KEYWORDS["ldl"],
    }
    DYSLIPIDEMIA_RANGES = {
        "total_cholesterol": (50, 600),
        "triglyceride": (20, 2000),
        "hdl": (10, 200),
        "ldl": (20, 500),
    }

    result = {k: None for k in DYSLIPIDEMIA_KEYWORDS}

    # ── 전략 1: 키워드 순서 탐지 후 뒤따르는 값 블록 매핑 ────────────────────
    keyword_order = []  # (field, line_index)
    for i, line in enumerate(lines):
        # 현재 줄 + 다음 줄 합쳐서 키워드 탐지 (고밀도\n콜레스테롤 같은 분리 대응)
        combined = line
        if i + 1 < len(lines):
            combined = line + " " + lines[i + 1]
        for field, kws in DYSLIPIDEMIA_KEYWORDS.items():
            if is_keyword_match(combined, kws) and field not in [f for f, _ in keyword_order]:
                keyword_order.append((field, i))

    if len(keyword_order) >= 2:
        last_kw_idx = keyword_order[-1][1]
        # 마지막 키워드 이후 최대 15줄에서 값/비해당 수집
        value_lines = lines[last_kw_idx + 1 : last_kw_idx + 16]
        collected = []  # (type, value) - type: "not_measured" or "number"
        for vl in value_lines:
            if is_not_measured(vl):
                collected.append(("not_measured", NOT_MEASURED_VALUE))
            else:
                for field, (low, high) in DYSLIPIDEMIA_RANGES.items():
                    nums = [n for n in extract_numbers(vl) if low <= n <= high]
                    if nums:
                        collected.append(("number", nums[0]))
                        break

            if len(collected) >= len(keyword_order):
                break

        for idx, (field, _) in enumerate(keyword_order):
            if idx < len(collected):
                result[field] = collected[idx][1]

    # ── 전략 2: 매핑 실패한 필드는 개별 키워드 파싱으로 fallback ─────────────
    for field, (low, high) in DYSLIPIDEMIA_RANGES.items():
        if result[field] is not None:
            continue
        for i, line in enumerate(lines):
            if not is_keyword_match(line, DYSLIPIDEMIA_KEYWORDS[field]):
                continue
            check = lines[i : i + 4]
            if any(is_not_measured(line) for line in check):
                result[field] = NOT_MEASURED_VALUE
                break
            for line in check:
                nums = [n for n in extract_numbers(line) if low <= n <= high]
                if nums:
                    result[field] = nums[0]
                    break
            if result[field] is not None:
                break

    return result


def extract_fields(lines: list[str]) -> dict:
    extracted = {k: None for k in FIELD_KEYS}

    systolic, diastolic = parse_blood_pressure(lines)
    if systolic and validate_value("systolic_bp", systolic):
        extracted["systolic_bp"] = systolic
    if diastolic and validate_value("diastolic_bp", diastolic):
        extracted["diastolic_bp"] = diastolic

    height, weight = parse_height_weight(lines)
    if height:
        extracted["height_cm"] = height
    if weight:
        extracted["weight_kg"] = weight

    bmi = parse_bmi(lines)
    if bmi is not None:
        extracted["bmi"] = bmi

    waist = parse_waist(lines)
    if waist is not None:
        extracted["waist_cm"] = waist

    glucose = parse_fasting_glucose(lines)
    if glucose is not None:
        extracted["fasting_glucose"] = glucose

    dyslipidemia = parse_dyslipidemia_all(lines)
    for field in ["total_cholesterol", "triglyceride", "hdl", "ldl"]:
        extracted[field] = dyslipidemia[field]

    return extracted


# ── CLOVA OCR API 호출 ────────────────────────────────────────────────────────


def call_clova_api(file_path: Path) -> list[str]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        request_json = {
            "images": [{"format": "pdf", "name": file_path.stem}],
            "requestId": str(uuid.uuid4()),
            "version": "V2",
            "timestamp": int(time.time() * 1000),
        }
        with open(file_path, "rb") as f:
            files = {
                "message": (None, json.dumps(request_json), "application/json"),
                "file": (file_path.name, f, "application/pdf"),
            }
            headers = {"X-OCR-SECRET": CLOVA_SECRET_KEY}
            response = requests.post(CLOVA_API_URL, headers=headers, files=files, timeout=30)
    else:
        request_json = {
            "images": [{"format": suffix.replace(".", ""), "name": file_path.stem}],
            "requestId": str(uuid.uuid4()),
            "version": "V2",
            "timestamp": int(time.time() * 1000),
        }
        with open(file_path, "rb") as f:
            files = {
                "message": (None, json.dumps(request_json), "application/json"),
                "file": (file_path.name, f, "image/jpeg"),
            }
            headers = {"X-OCR-SECRET": CLOVA_SECRET_KEY}
            response = requests.post(CLOVA_API_URL, headers=headers, files=files, timeout=30)

    if response.status_code != 200:
        raise Exception(f"CLOVA API 오류: {response.status_code} {response.text}")

    result = response.json()
    lines = []
    for image in result.get("images", []):
        for field in image.get("fields", []):
            text = clean_line(field.get("inferText", "").strip())
            if text:
                lines.append(text)

    return lines


# ── 정확도 측정 ───────────────────────────────────────────────────────────────


def evaluate(extracted: dict, subject: str) -> dict:
    """
    gt_val이 None인 필드(동욱님 BMI 등)는 total에서 제외하고
    match: None으로 표시한다.
    """
    gt = ground_truth["subjects"][subject]["ground_truth"]
    correct = 0
    total = 0
    details = {}

    for key in FIELD_KEYS:
        gt_val = gt.get(key)
        ext_val = extracted.get(key)

        # gt가 null → 정답 없음, 평가 제외
        if gt_val is None:
            details[key] = {"gt": None, "extracted": ext_val, "match": None}
            continue

        total += 1
        match = False

        if gt_val == NOT_MEASURED_VALUE:
            match = ext_val == NOT_MEASURED_VALUE
        elif ext_val is not None and ext_val != NOT_MEASURED_VALUE:
            tolerance = abs(float(gt_val)) * 0.05
            match = abs(float(ext_val) - float(gt_val)) <= tolerance

        if match:
            correct += 1
        details[key] = {"gt": gt_val, "extracted": ext_val, "match": match}

    accuracy = round(correct / total * 100, 1) if total > 0 else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy, "details": details}


# ── 결과 출력/저장 ────────────────────────────────────────────────────────────


def print_result(subject: str, file_type: str, eval_result: dict, elapsed_ms: int):
    print(f"\n{'=' * 50}")
    print(f"  {OCR_NAME.upper()} | {subject} | {file_type.upper()}")
    print("=" * 50)
    print(f"  정확도: {eval_result['correct']}/{eval_result['total']} ({eval_result['accuracy']}%)")
    print(f"  속도:   {elapsed_ms}ms")
    print("-" * 50)
    for key, detail in eval_result["details"].items():
        if detail["match"] is None:
            status = "⬜"
            note = "(정답 없음 - 평가 제외)"
        elif detail["match"]:
            status = "✅"
            note = ""
        else:
            status = "❌"
            note = ""
        ext = "비해당" if detail["extracted"] == NOT_MEASURED_VALUE else detail["extracted"]
        print(f"  {status} {key}: 정답={detail['gt']} / 추출={ext} {note}")
    print("=" * 50)


def save_result(subject: str, file_type: str, eval_result: dict, elapsed_ms: int):
    output = {
        "ocr_engine": OCR_NAME,
        "subject": subject,
        "file_type": file_type,
        "accuracy": eval_result["accuracy"],
        "correct": eval_result["correct"],
        "total": eval_result["total"],
        "elapsed_ms": elapsed_ms,
        "details": eval_result["details"],
    }
    filename = RESULTS_DIR / f"{OCR_NAME}_{subject}_{file_type}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {filename}")


# ── README 자동 생성 ──────────────────────────────────────────────────────────


def generate_readme(all_results: list):
    if not all_results:
        return

    jpg_results = [r for r in all_results if r["file_type"] == "jpg"]
    pdf_results = [r for r in all_results if r["file_type"] == "pdf"]

    avg_acc_jpg = (
        round(sum(r["eval_result"]["accuracy"] for r in jpg_results) / len(jpg_results), 1) if jpg_results else 0
    )
    avg_acc_pdf = (
        round(sum(r["eval_result"]["accuracy"] for r in pdf_results) / len(pdf_results), 1) if pdf_results else 0
    )
    avg_speed = round(sum(r["elapsed_ms"] for r in all_results) / len(all_results))
    subjects = list(dict.fromkeys(r["subject"] for r in all_results))

    jpg_rows = [
        f"| {r['subject']} | {r['eval_result']['correct']}/{r['eval_result']['total']} ({r['eval_result']['accuracy']}%) | {r['elapsed_ms']}ms |"
        for r in jpg_results
    ]
    pdf_rows = [
        f"| {r['subject']} | {r['eval_result']['correct']}/{r['eval_result']['total']} ({r['eval_result']['accuracy']}%) | {r['elapsed_ms']}ms |"
        for r in pdf_results
    ]

    col_header = f"| 컬럼 | {' | '.join(subjects)} | 평균 |"
    col_sep = f"|------|{'--------|' * len(subjects)}--------|"
    col_rows = []
    for key in FIELD_KEYS:
        cells = []
        for subj in subjects:
            r = next((x for x in jpg_results if x["subject"] == subj), None)
            if not r:
                cells.append("-")
                continue
            detail = r["eval_result"]["details"].get(key, {})
            if detail.get("match") is None:
                cells.append("⬜ 평가제외")
            elif detail["match"]:
                cells.append("✅ 비해당" if detail["extracted"] == NOT_MEASURED_VALUE else "✅")
            else:
                cells.append(f"❌ ({detail['extracted']})")
        match_count = sum(
            1
            for subj in subjects
            for r in jpg_results
            if r["subject"] == subj and r["eval_result"]["details"].get(key, {}).get("match") is True
        )
        col_rows.append(f"| {key} | {' | '.join(cells)} | {match_count}/{len(subjects)} |")

    now = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")

    content = f"""# Naver CLOVA OCR

## 테스트 결과 요약
- **평균 정확도 (JPG)**: {avg_acc_jpg}%
- **평균 정확도 (PDF)**: {avg_acc_pdf}%
- **평균 속도**: {avg_speed}ms
- **테스트 일시**: {now}

## 설치 방법
```bash
pip install requests python-dotenv
```

## 환경변수 (.env)
```
CLOVA_API_URL=https://...
CLOVA_SECRET_KEY=...
```

## JPG 테스트
| 대상 | 정확도 | 속도 |
|------|--------|------|
{chr(10).join(jpg_rows) if jpg_rows else "_해당 없음_"}

## PDF 테스트
| 대상 | 정확도 | 속도 |
|------|--------|------|
{chr(10).join(pdf_rows) if pdf_rows else "_해당 없음_"}

## 컬럼별 인식 결과 (JPG 기준)
{col_header}
{col_sep}
{chr(10).join(col_rows)}

## 범례
- ✅ 정답 일치
- ✅ 비해당: 비해당 정확히 인식
- ❌ (값): 오인식
- ⬜ 평가제외: ground_truth가 null (수치 없는 필드)

## 장점
- 한국어 특화 높은 인식률
- PDF 직접 지원
- 표 구조 인식 가능

## 단점
- 유료 API (월 1,000건 무료)
- 네트워크 필요

## 결론
- 추후 작성
"""

    readme_path = Path(__file__).parent / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n📝 README.md 자동 생성: {readme_path}")


# ── 테스트 실행 ───────────────────────────────────────────────────────────────


def run_test(file_path: Path, subject: str, file_type: str):
    print(f"\n분석 중: {file_path.name}")
    try:
        start = time.time()
        lines = call_clova_api(file_path)
        elapsed = round((time.time() - start) * 1000)
    except Exception as e:
        print(f"  ❌ API 오류: {e}")
        return None

    extracted = extract_fields(lines)
    eval_result = evaluate(extracted, subject)
    print_result(subject, file_type, eval_result, elapsed)
    save_result(subject, file_type, eval_result, elapsed)
    return {"subject": subject, "file_type": file_type, "eval_result": eval_result, "elapsed_ms": elapsed}


def main():
    print("🔍 Naver CLOVA OCR 테스트 시작 (이미지 + PDF)")
    print(f"이미지 경로: {IMAGES_DIR}\n")

    if not CLOVA_API_URL or not CLOVA_SECRET_KEY:
        print("❌ .env에 CLOVA_API_URL과 CLOVA_SECRET_KEY를 입력해주세요.")
        return

    if not IMAGES_DIR.exists():
        print(f"❌ 이미지 폴더가 없습니다: {IMAGES_DIR}")
        return

    all_results = []
    files = sorted(IMAGES_DIR.iterdir())

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix not in [".jpg", ".jpeg", ".png", ".pdf"]:
            print(f"⏭️  스킵: {file_path.name}")
            continue

        # ground_truth subject 매핑
        subject = None
        for name in ground_truth["subjects"]:
            if name in file_path.name:
                subject = name
                break

        if not subject:
            print(f"⚠️  대상자 매핑 실패: {file_path.name}")
            continue

        file_type = "pdf" if suffix == ".pdf" else "jpg"
        result = run_test(file_path, subject, file_type)
        if result:
            all_results.append(result)

    generate_readme(all_results)
    print("\n✅ 테스트 완료")


if __name__ == "__main__":
    main()
