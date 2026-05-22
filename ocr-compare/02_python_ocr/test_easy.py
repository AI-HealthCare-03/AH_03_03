"""
02_python_ocr/test_easy.py

EasyOCR 기반 건강검진표 수치 추출 테스트
이미지(JPG) + PDF 지원

실행: python test_easy.py
"""

import json
import re
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
from pdf2image import convert_from_path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────

IMAGES_DIR = Path(__file__).parent.parent / "results" / "images" / "checkup"
RESULTS_DIR = Path(__file__).parent.parent / "results"
COLUMNS_FILE = RESULTS_DIR / "columns.json"
GROUND_TRUTH_FILE = RESULTS_DIR / "ground_truth.json"
POPPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin"
OCR_NAME = "easyocr"

# ── 데이터 로드 ───────────────────────────────────────────────────────────────

with open(COLUMNS_FILE, encoding="utf-8") as f:
    columns_data = json.load(f)

with open(GROUND_TRUTH_FILE, encoding="utf-8") as f:
    ground_truth = json.load(f)

NOT_MEASURED_VALUE = columns_data["not_measured_value"]
NOT_MEASURED_KEYWORDS = columns_data["not_measured_keywords"]

NULLABLE_FIELDS = set()
FIELD_KEYS = []
for category in columns_data["columns"].values():
    for field in category["fields"]:
        FIELD_KEYS.append(field["key"])
        if field.get("nullable"):
            NULLABLE_FIELDS.add(field["key"])

# ── EasyOCR 싱글톤 ────────────────────────────────────────────────────────────

_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        print("EasyOCR 모델 로딩 중...")
        _ocr_reader = easyocr.Reader(["ko", "en"], gpu=False)
        print("EasyOCR 모델 로딩 완료\n")
    return _ocr_reader


# ── 키워드 매핑 ───────────────────────────────────────────────────────────────

FIELD_KEYWORDS = {
    "systolic_bp": ["수축기", "수축기혈압", "최고혈압", "SBP", "mmHg"],
    "diastolic_bp": ["이완기", "이완기혈압", "최저혈압", "DBP"],
    "fasting_glucose": ["공복혈당", "공복 혈당", "당뇨병", "GLU"],
    "total_cholesterol": ["총콜레스테롤", "총 콜레스테롤", "TC", "T-CHO"],
    "triglyceride": ["중성지방", "TG"],
    "hdl": ["고밀도 콜레스테롤", "고밀도콜레스테롤", "HDL"],
    "ldl": ["저밀도 콜레스테롤", "저밀도콜레스테롤", "LDL"],
    "height_cm": ["키", "신장", "Height"],
    "weight_kg": ["체중", "몸무게", "Weight"],
    "bmi": ["체질량지수", "BMI", "비만도"],
    "waist_cm": ["허리둘레", "허리 둘레"],
}

# ── 노이즈 패턴 ───────────────────────────────────────────────────────────────

NOISE_PATTERN = re.compile(r"\d+\.?\d*\s*(만만|미만|이하|이상|초과|이내)")
CHECKBOX_PATTERN = re.compile(r"[■□▣▪●○◆◇]")


def clean_line(text):
    text = CHECKBOX_PATTERN.sub("", text)
    text = NOISE_PATTERN.sub("", text)
    return text.strip()


def extract_numbers(text):
    cleaned = clean_line(text).replace(",", "")
    matches = re.findall(r"\d+\.?\d*", cleaned)
    return [float(m) for m in matches if m]


def is_keyword_match(text, keywords):
    upper = text.upper()
    return any(kw.upper() in upper for kw in keywords)


def is_not_measured(text):
    return any(kw.upper() in text.upper() for kw in NOT_MEASURED_KEYWORDS)


def validate_value(field, value):
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


def is_reference_line(line):
    """참고치 줄 여부 확인 (전단계, 의심, 이상, 또는 등 포함)."""
    ref_keywords = ["전단계", "의심", "또는", "이하", "이상이", "고혈압의심"]
    return any(kw in line for kw in ref_keywords)


def parse_blood_pressure(lines):
    for line in lines:
        if is_keyword_match(line, ["수축기", "mmHg", "SBP"]):
            nums = [n for n in extract_numbers(line) if 40 <= n <= 250]
            if len(nums) >= 2:
                return nums[0], nums[1]

    for i, line in enumerate(lines):
        if "mmHg" in line or "mHg" in line or "mmhg" in line.lower():
            context = lines[max(0, i - 3) : i + 3]
            combined = " ".join(context)
            nums = [n for n in extract_numbers(combined) if 40 <= n <= 250]
            if len(nums) >= 2:
                return nums[0], nums[1]
            if len(nums) == 1:
                return nums[0], None

    return None, None


def parse_height_weight(lines):
    for i, line in enumerate(lines):
        if is_keyword_match(line, ["키", "신장", "몸무게", "체중"]) or ("키" in line and "몸무게" in line):
            context = lines[i : i + 3]
            combined = " ".join(context)
            nums = extract_numbers(combined)
            for j in range(len(nums) - 1):
                h, w = nums[j], nums[j + 1]
                if 100 <= h <= 250 and 20 <= w <= 300:
                    return h, w
    return None, None


def parse_bmi(lines):
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
            next_nums = [n for n in extract_numbers(lines[i + 1]) if 10 <= n <= 50]
            if next_nums:
                return next_nums[0]
    return None


def parse_waist(lines):
    for i, line in enumerate(lines):
        if not is_keyword_match(line, FIELD_KEYWORDS["waist_cm"]):
            continue
        nums = [n for n in extract_numbers(line) if 40 <= n <= 200]
        if nums:
            return nums[0]
        if i + 1 < len(lines):
            next_nums = [n for n in extract_numbers(lines[i + 1]) if 40 <= n <= 200]
            if next_nums:
                return next_nums[0]
    return None


def parse_fasting_glucose(lines):
    for i, line in enumerate(lines):
        if not is_keyword_match(line, FIELD_KEYWORDS["fasting_glucose"]):
            continue
        nums = [n for n in extract_numbers(line) if 40 <= n <= 600]
        if nums:
            return nums[0]
        if i + 1 < len(lines):
            next_nums = [n for n in extract_numbers(lines[i + 1]) if 40 <= n <= 600]
            if next_nums:
                return next_nums[0]
    return None


def parse_dyslipidemia(lines, field):
    ranges = {
        "total_cholesterol": (50, 600),
        "triglyceride": (20, 2000),
        "hdl": (10, 200),
        "ldl": (20, 500),
    }
    low, high = ranges[field]
    for i, line in enumerate(lines):
        if not is_keyword_match(line, FIELD_KEYWORDS[field]):
            continue
        check = [line]
        if i + 1 < len(lines):
            check.append(lines[i + 1])
        if i + 2 < len(lines):
            check.append(lines[i + 2])
        if any(is_not_measured(line) for line in check):
            return NOT_MEASURED_VALUE
        for line in check:
            nums = [n for n in extract_numbers(line) if low <= n <= high]
            if nums:
                return nums[0]
    return None


def extract_fields(lines):
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

    for field in ["total_cholesterol", "triglyceride", "hdl", "ldl"]:
        extracted[field] = parse_dyslipidemia(lines, field)

    return extracted


# ── OCR 실행 ──────────────────────────────────────────────────────────────────


def run_ocr_on_image(img):
    """OpenCV 이미지에서 EasyOCR 실행 후 텍스트 라인 반환."""
    reader = get_ocr_reader()
    results = reader.readtext(img)
    lines = []
    for _, text, _ in results:
        cleaned = clean_line(text.strip())
        if cleaned:
            lines.append(cleaned)
    return lines


def run_ocr_on_file(file_path):
    """이미지 파일에서 OCR 실행."""
    arr = np.fromfile(str(file_path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return run_ocr_on_image(img)


def pdf_to_lines(file_path):
    """PDF를 이미지로 변환 후 OCR 실행하여 텍스트 라인 반환."""
    import io

    print("  PDF 변환 중...")
    images = convert_from_path(
        str(file_path),
        dpi=200,
        poppler_path=POPPLER_PATH,
    )
    all_lines = []
    for i, img in enumerate(images):
        print(f"  페이지 {i + 1}/{len(images)} OCR 실행 중...")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        lines = run_ocr_on_image(cv_img)
        all_lines.extend(lines)
    return all_lines


# ── 정확도 측정 ───────────────────────────────────────────────────────────────


def evaluate(extracted, subject):
    gt = ground_truth["subjects"][subject]["ground_truth"]
    correct = 0
    total = 0
    details = {}

    for key in FIELD_KEYS:
        gt_val = gt.get(key)
        ext_val = extracted.get(key)

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


def print_result(subject, file_type, eval_result, elapsed_ms):
    print(f"\n{'=' * 50}")
    print(f"  {OCR_NAME.upper()} | {subject} | {file_type}")
    print("=" * 50)
    print(f"  정확도: {eval_result['correct']}/{eval_result['total']} ({eval_result['accuracy']}%)")
    print(f"  속도:   {elapsed_ms}ms")
    print("-" * 50)
    for key, detail in eval_result["details"].items():
        status = "⬜" if detail["match"] is None else "✅" if detail["match"] else "❌"
        ext = "비해당" if detail["extracted"] == NOT_MEASURED_VALUE else detail["extracted"]
        print(f"  {status} {key}: 정답={detail['gt']} / 추출={ext}")
    print("=" * 50)


def save_result(subject, file_type, eval_result, elapsed_ms):
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


def generate_readme(all_results):
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
                cells.append("⬜")
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

    content = f"""# EasyOCR 1.7.2

## 테스트 결과 요약
- **평균 정확도 (JPG)**: {avg_acc_jpg}%
- **평균 정확도 (PDF)**: {avg_acc_pdf}%
- **평균 속도**: {avg_speed}ms
- **테스트 일시**: {now}

## 설치 방법
```bash
pip install easyocr pdf2image
# poppler 별도 설치 필요
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
- ✅ 비해당 정확히 인식
- ❌ 오인식
- ⬜ 정답 데이터 없음

## 장점
- 설치 간단
- 한국어/영어 혼합 인식 양호
- GPU 없이도 동작

## 단점
- PaddleOCR 대비 속도 느림
- 한영 혼합 단위 오인식
- 초기 모델 다운로드 필요

## 결론
- 추후 작성
"""

    readme_path = Path(__file__).parent / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n📝 README.md 자동 생성: {readme_path}")


# ── 테스트 실행 ───────────────────────────────────────────────────────────────


def run_test(file_path, subject, file_type):
    print(f"\n분석 중: {file_path.name}")
    start = time.time()
    lines = run_ocr_on_file(file_path)
    elapsed = round((time.time() - start) * 1000)

    extracted = extract_fields(lines)
    eval_result = evaluate(extracted, subject)
    print_result(subject, file_type, eval_result, elapsed)
    save_result(subject, file_type, eval_result, elapsed)
    return {"subject": subject, "file_type": file_type, "eval_result": eval_result, "elapsed_ms": elapsed}


def run_pdf_test(file_path, subject):
    print(f"\n분석 중 (PDF): {file_path.name}")
    start = time.time()
    lines = pdf_to_lines(file_path)
    elapsed = round((time.time() - start) * 1000)

    extracted = extract_fields(lines)
    eval_result = evaluate(extracted, subject)
    print_result(subject, "pdf", eval_result, elapsed)
    save_result(subject, "pdf", eval_result, elapsed)
    return {"subject": subject, "file_type": "pdf", "eval_result": eval_result, "elapsed_ms": elapsed}


def main():
    print("🔍 EasyOCR 테스트 시작 (이미지 + PDF)")
    print(f"이미지 경로: {IMAGES_DIR}\n")

    all_results = []
    files = sorted(IMAGES_DIR.iterdir())

    for file_path in files:
        suffix = file_path.suffix.lower()

        subject = None
        for name in ground_truth["subjects"]:
            if name in file_path.name:
                subject = name
                break

        if not subject:
            print(f"⚠️  대상자 매핑 실패: {file_path.name}")
            continue

        if suffix == ".pdf":
            result = run_pdf_test(file_path, subject)
        elif suffix in [".jpg", ".jpeg", ".png"]:
            result = run_test(file_path, subject, suffix.replace(".", ""))
        else:
            print(f"⏭️  스킵: {file_path.name}")
            continue

        all_results.append(result)

    generate_readme(all_results)
    print("\n✅ 테스트 완료")


if __name__ == "__main__":
    main()
