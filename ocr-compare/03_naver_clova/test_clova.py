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

load_dotenv(Path(__file__).parent.parent.parent / ".env")

CLOVA_API_URL = os.getenv("CLOVA_API_URL", "")
CLOVA_SECRET_KEY = os.getenv("CLOVA_SECRET_KEY", "")

# ── 경로 설정 ─────────────────────────────────────────────────────────────────

IMAGES_DIR = Path(__file__).parent.parent / "results" / "images" / "checkup"
RESULTS_DIR = Path(__file__).parent.parent / "results"
COLUMNS_FILE = RESULTS_DIR / "columns.json"
GROUND_TRUTH_FILE = RESULTS_DIR / "ground_truth.json"
OCR_NAME = "clova"

# ── CLOVA OCR API 설정 ────────────────────────────────────────────────────────
# 네이버 클라우드 콘솔 → CLOVA OCR → 도메인 → API 탭에서 확인

CLOVA_API_URL = "여기에_invoke_url_입력"
CLOVA_SECRET_KEY = "여기에_secret_key_입력"

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


def parse_blood_pressure(lines):
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


# ── CLOVA OCR API 호출 ────────────────────────────────────────────────────────


def call_clova_api(file_path):
    """
    CLOVA OCR API 호출.
    이미지 또는 PDF 파일을 전송하고 인식된 텍스트 라인을 반환합니다.
    """
    suffix = file_path.suffix.lower()

    # MIME 타입 설정
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".pdf": "application/pdf",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    # 요청 메시지 구성
    request_body = {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "lang": "ko",
        "images": [
            {
                "format": suffix.replace(".", ""),
                "name": file_path.name,
            }
        ],
        "enableTableDetection": False,
    }

    with open(file_path, "rb") as f:
        files = {
            "message": (None, json.dumps(request_body), "application/json"),
            "file": (file_path.name, f, mime_type),
        }
        headers = {"X-OCR-SECRET": CLOVA_SECRET_KEY}
        response = requests.post(
            CLOVA_API_URL,
            headers=headers,
            files=files,
            timeout=30,
        )

    if response.status_code != 200:
        raise Exception(f"CLOVA API 오류: {response.status_code} {response.text}")

    result = response.json()

    # 텍스트 라인 추출
    lines = []
    for image in result.get("images", []):
        for field in image.get("fields", []):
            text = clean_line(field.get("inferText", "").strip())
            if text:
                lines.append(text)

    return lines


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

    content = f"""# Naver CLOVA OCR

## 테스트 결과 요약
- **평균 정확도 (JPG)**: {avg_acc_jpg}%
- **평균 정확도 (PDF)**: {avg_acc_pdf}%
- **평균 속도**: {avg_speed}ms
- **테스트 일시**: {now}

## 설치 방법
```bash
pip install requests
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


def run_test(file_path, subject, file_type):
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

    # API 키 확인
    if "여기에" in CLOVA_API_URL or "여기에" in CLOVA_SECRET_KEY:
        print("❌ CLOVA_API_URL과 CLOVA_SECRET_KEY를 입력해주세요.")
        return

    all_results = []
    files = sorted(IMAGES_DIR.iterdir())

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix not in [".jpg", ".jpeg", ".png", ".pdf"]:
            print(f"⏭️  스킵: {file_path.name}")
            continue

        subject = None
        for name in ground_truth["subjects"]:
            if name in file_path.name:
                subject = name
                break

        if not subject:
            print(f"⚠️  대상자 매핑 실패: {file_path.name}")
            continue

        file_type = suffix.replace(".", "")
        result = run_test(file_path, subject, file_type)
        if result:
            all_results.append(result)

    generate_readme(all_results)
    print("\n✅ 테스트 완료")


if __name__ == "__main__":
    main()
