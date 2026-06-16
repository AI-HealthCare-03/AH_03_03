from __future__ import annotations

import json
from pathlib import Path

from ai_runtime.llm.rag.source_loader import DEFAULT_RAG_SOURCE_DIR, load_rag_source_index
from ai_runtime.llm.rag_sources import is_allowed_domain, is_allowed_rag_source

REQUIRED_SOURCE_METADATA_FIELDS = {
    "id",
    "disease_code",
    "title",
    "filename",
    "source_org",
    "source_url",
    "source_type",
    "source_trust_level",
    "topic_tags",
    "issue_keys",
    "usage_scope",
    "review_status",
    "enabled",
}


def test_enabled_rag_sources_have_required_metadata_fields() -> None:
    index_path = Path(DEFAULT_RAG_SOURCE_DIR) / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    enabled_sources = [source for source in payload if source.get("enabled") is True]

    assert enabled_sources
    for source in enabled_sources:
        missing_fields = REQUIRED_SOURCE_METADATA_FIELDS - set(source)
        assert missing_fields == set(), f"{source.get('id')} missing {sorted(missing_fields)}"
        assert str(source["source_org"]).strip()
        assert str(source["source_url"]).startswith("https://")
        assert str(source["source_type"]).strip()
        assert str(source["source_trust_level"]).strip()


def test_dyslipidemia_source_metadata_is_official_guideline_candidate() -> None:
    metadata_by_id = {metadata.id: metadata for metadata in load_rag_source_index()}
    dyslipidemia = metadata_by_id["dyslipidemia"]

    assert dyslipidemia.disease_code == "DL"
    assert dyslipidemia.source_org == "대한지질·동맥경화학회, 질병관리청 국가건강정보포털"
    assert dyslipidemia.source_url.startswith("https://lipid.or.kr/")
    assert dyslipidemia.source_type == "clinical_guideline"
    assert dyslipidemia.source_trust_level == "official_guideline"
    assert {"포화지방", "트랜스지방", "튀김", "가공식품", "콜레스테롤"}.issubset(set(dyslipidemia.topic_tags))
    assert {"fat_high", "fiber_support", "cholesterol_management"}.issubset(set(dyslipidemia.issue_keys))


def test_lipid_domain_is_allowed_for_defined_official_society_source() -> None:
    assert is_allowed_domain("https://lipid.or.kr/reference/guideline.php?idx=1281") is True
    assert is_allowed_rag_source(
        source_name="대한지질·동맥경화학회",
        url="https://lipid.or.kr/reference/guideline.php?idx=1281",
    )


def test_ckd_source_skeleton_metadata_is_safe_and_disabled() -> None:
    metadata_by_id = {metadata.id: metadata for metadata in load_rag_source_index()}
    ckd = metadata_by_id["ckd"]
    content = (Path(DEFAULT_RAG_SOURCE_DIR) / ckd.filename).read_text(encoding="utf-8")

    assert ckd.enabled is False
    assert ckd.disease_code == "CKD"
    assert ckd.review_status == "candidate_unreviewed"
    assert ckd.safety_level == "high_caution"
    assert ckd.source_type == "clinical_guideline"
    assert ckd.source_trust_level == "official_guideline"
    assert "kidney_caution" in ckd.issue_keys
    assert "구체적인 제한량" in content
    assert "의료진과 상담" in content
