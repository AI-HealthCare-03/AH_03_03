"""
experiment/ai/ocr/checkup/eval/evaluate_checkup_vision.py

GPT Vision 기반 건강검진표 이미지 추출 정확도 평가 스크립트

실행 방법:
    cd experiment
    python -m ai.ocr.checkup.eval.evaluate_checkup_vision

폴더 구조:
    experiment/ai/ocr/checkup/eval/
    ├── images/                   # health_check_00001.jpg ~
    ├── ground_truth_10000.json
    ├── results/
    └── evaluate_checkup_vision.py

비고:
    - 정확도 비교는 12개 필드 기준 (BMI 포함)
    - BMI 추출값이 null이면 신장+몸무게로 계산하여 [계산값] 표기 (구형 검진표 대응)
    - 비해당은 문자열 일치로 정답 처리
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# experiment/.env 로드
load_dotenv(Path(__file__).parents[4] / ".env")

from ai.cv.providers.gpt_vision import AnalysisType, VisionClient

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
GT_PATH = BASE_DIR / "ground_truth_10000.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 정확도 비교 대상 12개 필드
# BMI: 이미지에서 직접 추출, 구형 검진표처럼 없는 경우 신장+몸무게로 계산
ACCURACY_FIELDS = [
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "hb",
    "total_cholesterol",
    "triglyceride",
    "hdl",
    "ldl",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
]


def calc_bmi(height_cm, weight_kg) -> float | None:
    """신장(cm) + 몸무게(kg)로 BMI 계산 (소수점 1자리)"""
    try:
        h = float(height_cm) / 100
        w = float(weight_kg)
        if h <= 0:
            return None
        return round(w / (h * h), 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def normalize(value) -> str | None:
    if value is None:
        return None
    return str(value).strip().replace(" ", "")


def is_match(gt_val, vision_val) -> bool | None:
    gt_norm = normalize(gt_val)
    if gt_norm is None:
        return None
    if gt_norm == "비해당":
        return normalize(vision_val) == "비해당"
    if vision_val is None:
        return False
    try:
        return float(gt_norm) == float(normalize(vision_val))
    except (ValueError, TypeError):
        return gt_norm == normalize(vision_val)


async def evaluate():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. experiment/.env 파일을 확인하세요.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    client = VisionClient(api_key=api_key, model=model)

    # GT 로드 및 역매핑 생성: pdf_stem → (subject_name, gt_dict)
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    stem_to_gt: dict[str, tuple[str, dict]] = {}
    for name, info in gt_data["subjects"].items():
        pdf_name = info["files"].get("health_check_pdf")
        if pdf_name:
            stem = Path(pdf_name).stem  # "health_check_00001"
            stem_to_gt[stem] = (name, info["ground_truth"])

    # 이미지 목록
    images = sorted(IMAGES_DIR.glob("health_check_*.jpg"))
    total = len(images)

    print(f"\n{'='*60}")
    print(f"  GPT Vision 건강검진 이미지 평가 | {total}건")
    print(f"  모델: {model}")
    print(f"{'='*60}")

    all_results = []
    accuracies = []
    failed = []

    for i, img_path in enumerate(images, 1):
        stem = img_path.stem  # "health_check_00001"

        if stem not in stem_to_gt:
            failed.append({"파일": img_path.name, "이유": "GT 매핑 없음"})
            continue

        subject_name, gt = stem_to_gt[stem]

        try:
            image_bytes = img_path.read_bytes()
            response = await client.analyze(AnalysisType.CHECKUP, image_bytes, "image/jpeg")
            extracted: dict = response.get("extracted_data", {})

            # BMI null이면 신장+몸무게로 계산
            bmi_source = "extracted"
            if extracted.get("bmi") is None:
                calc = calc_bmi(extracted.get("height_cm"), extracted.get("weight_kg"))
                if calc is not None:
                    extracted["bmi"] = calc
                    bmi_source = "calculated"

            # 12개 필드 정확도 계산 (BMI 포함)
            correct = 0
            total_fields = 0
            detail = {}

            for field in ACCURACY_FIELDS:
                gt_val = gt.get(field)
                vision_val = extracted.get(field)
                result = is_match(gt_val, vision_val)
                if result is None:
                    detail[field] = f"제외 (GT={gt_val})"
                    continue
                total_fields += 1
                if result:
                    correct += 1
                    suffix = f" [계산값]" if field == "bmi" and bmi_source == "calculated" else ""
                    detail[field] = f"✅ (GT={gt_val}, Vision={vision_val}){suffix}"
                else:
                    suffix = f" [계산값]" if field == "bmi" and bmi_source == "calculated" else ""
                    detail[field] = f"❌ (GT={gt_val}, Vision={vision_val}){suffix}"

            acc = round(correct / total_fields, 4) if total_fields > 0 else 0.0
            accuracies.append(acc)

            all_results.append({
                "이름": subject_name,
                "파일": img_path.name,
                "정확도": round(acc * 100, 1),
                "정답수": correct,
                "전체수": total_fields,
                "analysis_status": response.get("analysis_status"),
                "상세": detail,
            })

        except Exception as e:
            failed.append({"파일": img_path.name, "이유": str(e)})

        if i % 10 == 0 or i == total:
            avg = round(sum(accuracies) / len(accuracies) * 100, 1) if accuracies else 0
            print(f"  진행: {i}/{total} | 현재 평균: {avg}%")

    avg = round(sum(accuracies) / len(accuracies) * 100, 1) if accuracies else 0

    print(f"\n{'='*60}")
    print(f"  평가 완료")
    print(f"  평균 정확도: {avg}% ({len(accuracies)}건 평가 / {len(failed)}건 실패)")
    print(f"  * 정확도는 12개 필드 기준 (BMI 포함, null 시 신장+몸무게 계산)")
    print(f"{'='*60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"{timestamp}_vision_eval.json"

    summary = {
        "평가일시": datetime.now().isoformat(),
        "평가방식": f"GPT Vision ({model})",
        "평가건수": len(accuracies),
        "실패건수": len(failed),
        "평균정확도": f"{avg}%",
        "비고": "정확도는 12개 필드 기준 (BMI 포함 — null 시 신장+몸무게로 계산, 상세에 [계산값] 표기)",
        "실패목록": failed,
        "상세결과": all_results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  💾 결과 저장: results/{result_path.name}\n")


if __name__ == "__main__":
    asyncio.run(evaluate())
