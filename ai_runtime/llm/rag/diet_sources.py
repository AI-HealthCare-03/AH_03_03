from __future__ import annotations

from pathlib import Path

from ai_runtime.llm.rag.source_loader import DEFAULT_RAG_SOURCE_DIR, load_rag_source_index

ISSUE_TO_RAG_CODES = {
    "sodium_high": ["HTN", "DIET_NUTRITION"],
    "carbohydrate_high": ["DM", "DIET_NUTRITION"],
    "sugar_high": ["DM", "FL", "DIET_NUTRITION"],
    "fat_high": ["DL", "FL", "DIET_NUTRITION"],
    "calorie_high": ["OBE", "FL", "DIET_NUTRITION"],
    "protein_support": ["ANEM", "DIET_NUTRITION"],
    "iron_support": ["ANEM", "DIET_NUTRITION"],
    "fiber_support": ["DL", "DM", "DIET_NUTRITION"],
    "kidney_caution": ["CKD", "DIET_CAUTION"],
    "late_night_or_irregular": ["OBE", "FL", "DIET_NUTRITION"],
    "alcohol_liver_support": ["FL", "DIET_CAUTION"],
}

DIET_RAG_QUERY_TEMPLATES = {
    "HTN": "고혈압 혈압 관리 저염 나트륨 국물 소스 식습관",
    "DM": "당뇨 혈당 관리 탄수화물 당류 식사요법 식이섬유",
    "DL": "이상지질혈증 콜레스테롤 중성지방 포화지방 튀김 식이섬유",
    "OBE": "비만 체중관리 열량 야식 폭식 간식 식사습관",
    "FL": "지방간 간기능 음주 당류 지방 체중관리 식사습관",
    "CKD": "신장기능 eGFR 요단백 식사 주의 의료진 상담 개인별 제한",
    "ANEM": "빈혈 혈색소 철분 단백질 비타민C 식사",
    "DIET_NUTRITION": "건강한 식생활 영양 균형 저염 저당 포화지방 식이섬유",
    "DIET_CAUTION": "질환자 식단 주의 진단 처방 금지 의료진 상담 제한식 개인별",
    "DIET_FAQ": "질환별 식단 질문 혈압 혈당 콜레스테롤 간수치 신장기능",
}


def enabled_rag_codes(source_dir: Path = DEFAULT_RAG_SOURCE_DIR) -> set[str]:
    return {metadata.disease_code for metadata in load_rag_source_index(source_dir) if metadata.enabled}


def resolve_diet_rag_codes(
    *,
    issue_keys: list[str] | tuple[str, ...],
    disease_codes: list[str] | tuple[str, ...] = (),
    source_dir: Path = DEFAULT_RAG_SOURCE_DIR,
) -> list[str]:
    enabled_codes = enabled_rag_codes(source_dir)
    candidates: list[str] = []
    candidates.extend(str(code).upper() for code in disease_codes)
    for issue_key in issue_keys:
        candidates.extend(ISSUE_TO_RAG_CODES.get(issue_key, ()))
    return _dedupe([code for code in candidates if code in enabled_codes])


def build_diet_rag_query(
    *,
    issue_keys: list[str] | tuple[str, ...],
    disease_codes: list[str] | tuple[str, ...] = (),
    source_dir: Path = DEFAULT_RAG_SOURCE_DIR,
) -> str:
    codes = resolve_diet_rag_codes(issue_keys=issue_keys, disease_codes=disease_codes, source_dir=source_dir)
    return " ".join(DIET_RAG_QUERY_TEMPLATES[code] for code in codes if code in DIET_RAG_QUERY_TEMPLATES)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
