"""
식단 이미지 검증용 샘플 추출 + 정답 CSV 통합 스크립트

기능:
1) [라벨]음식XXX_Val_json/<음식명>_json/ 폴더마다 json 파일을 최대 N개(기본 3개)씩 선택
2) 각 json에서 정답(Code Name=이미지 파일명, Name=정답 음식명)을 추출
3) 짝이 되는 [원천]음식XXX_Val/<음식명>/ 폴더에서 동일 파일명의 이미지를 찾아 매칭
4) 선택된 이미지를 출력 폴더로 복사하고, 정답을 하나의 CSV로 통합 저장

사용법:
    python build_validation_set.py \
        --validation-root "C:\\Users\\82106\\Desktop\\식단 검증 이미지\\Validation" \
        --output-dir "C:\\Users\\82106\\Desktop\\PycharmProjects\\AH_03_03\\ai_runtime\\cv\\food\\eval" \
        --samples-per-food 3

출력:
    <output-dir>/diet_ground_truth.csv   (컬럼: category, food_name, code_name, image_path, json_path)
    <output-dir>/sample_images/<음식명>/<파일명>.jpg  (검증용으로 복사된 샘플 이미지)
"""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


def find_label_source_pairs(validation_root: Path):
    """[라벨]음식XXX_Val_json <-> [원천]음식XXX_Val 폴더 쌍을 찾는다.

    주의: pathlib.glob()에서 "[...]"는 glob 문법상 문자 집합으로 해석되므로
    "[라벨]", "[원천]" 같은 실제 폴더명 접두사와는 매칭되지 않는다.
    따라서 glob 대신 폴더명 문자열을 직접 비교한다.
    """
    pairs = []
    candidates = [d for d in sorted(validation_root.iterdir()) if d.is_dir()]

    for label_dir in candidates:
        name = label_dir.name
        if not (name.startswith("[라벨]") and name.endswith("_Val_json")):
            continue

        # "[라벨]음식001_Val_json" -> "[원천]음식001_Val"
        suffix = name[len("[라벨]"):-len("_json")]  # "음식001_Val"
        source_dir = validation_root / f"[원천]{suffix}"

        if source_dir.exists():
            pairs.append((label_dir, source_dir))
        else:
            print(f"[경고] 짝이 되는 원천 폴더를 찾지 못했습니다: {source_dir}")

    return pairs


def collect_samples(label_dir: Path, source_dir: Path, samples_per_food: int):
    """카테고리(라벨) 폴더 하나에서 음식명별로 N개의 샘플(json+이미지+정답)을 수집한다."""
    rows = []
    copy_jobs = []  # (src_image_path, dest_relative_path)

    for food_json_dir in sorted(label_dir.iterdir()):
        if not food_json_dir.is_dir():
            continue

        # 폴더명 접미사 표기가 카테고리마다 다름: "가리비_json"(밑줄) / "가리비 json"(공백) 등
        # -> 끝에 붙은 구분자(_ 또는 공백) + "json"을 대소문자 무시하고 제거
        food_name_kor = re.sub(r"[_\s]*json$", "", food_json_dir.name, flags=re.IGNORECASE).strip()
        food_image_dir = source_dir / food_name_kor

        if not food_image_dir.exists():
            # 그래도 못 찾으면, 원천 폴더 목록에서 이름이 가장 비슷한 것을 찾아본다
            # (예: 양쪽 폴더명에 trailing space, 자모 정규화(NFC/NFD) 차이가 있는 경우 대비)
            candidates = {d.name.strip(): d for d in source_dir.iterdir() if d.is_dir()}
            food_image_dir = candidates.get(food_name_kor.strip())

        if food_image_dir is None or not food_image_dir.exists():
            print(f"[경고] 짝이 되는 이미지 폴더 없음: {source_dir / food_name_kor} "
                  f"(원본 라벨 폴더명: '{food_json_dir.name}')")
            continue

        json_files = sorted(food_json_dir.glob("*.json"))[:samples_per_food]
        for json_path in json_files:
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[경고] json 파싱 실패: {json_path} ({e})")
                continue

            if not data:
                continue

            # 같은 이미지의 entry는 Code Name / Name이 동일하므로 첫 entry만 사용
            entry = data[0]
            code_name = entry.get("Code Name", "").strip()
            answer_name = entry.get("Name", "").strip()
            if not code_name:
                print(f"[경고] Code Name 없음: {json_path}")
                continue

            image_path = food_image_dir / code_name
            if not image_path.exists():
                print(f"[경고] 매칭되는 이미지 없음: {image_path}")
                continue

            dest_rel = Path(food_name_kor) / code_name
            rows.append({
                "category": label_dir.name,
                "food_name_kor": food_name_kor,
                "answer_name": answer_name,
                "code_name": code_name,
                "image_path": str(image_path),
                "json_path": str(json_path),
                "sample_image_path": str(Path("sample_images") / dest_rel),
            })
            copy_jobs.append((image_path, dest_rel))

    return rows, copy_jobs


def main():
    parser = argparse.ArgumentParser(description="식단 검증 샘플 추출 및 정답 CSV 생성")
    parser.add_argument("--validation-root", required=True, help="Validation 폴더 경로 (예: 바탕화면/식단 검증 이미지/Validation)")
    parser.add_argument("--output-dir", required=True, help="결과(CSV, 샘플 이미지)를 저장할 폴더")
    parser.add_argument("--samples-per-food", type=int, default=3, help="음식명별로 추출할 샘플 수 (기본 3)")
    args = parser.parse_args()

    validation_root = Path(args.validation_root)
    output_dir = Path(args.output_dir)
    sample_image_root = output_dir / "sample_images"
    csv_path = output_dir / "diet_ground_truth.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_image_root.mkdir(parents=True, exist_ok=True)

    pairs = find_label_source_pairs(validation_root)
    if not pairs:
        print("[오류] [라벨]/[원천] 폴더 쌍을 찾지 못했습니다. --validation-root 경로를 확인하세요.")
        return

    all_rows = []
    all_copy_jobs = []
    for label_dir, source_dir in pairs:
        print(f"\n처리 중: {label_dir.name}  <->  {source_dir.name}")
        rows, copy_jobs = collect_samples(label_dir, source_dir, args.samples_per_food)
        print(f"  -> {len(rows)}개 샘플 수집")
        all_rows.extend(rows)
        all_copy_jobs.extend(copy_jobs)

    # 이미지 복사
    for src, dest_rel in all_copy_jobs:
        dest_path = sample_image_root / dest_rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_path)

    # CSV 저장
    fieldnames = ["category", "food_name_kor", "answer_name", "code_name",
                  "image_path", "json_path", "sample_image_path"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n완료: 총 {len(all_rows)}개 샘플")
    print(f"  - 정답 CSV: {csv_path}")
    print(f"  - 샘플 이미지: {sample_image_root}")


if __name__ == "__main__":
    main()
