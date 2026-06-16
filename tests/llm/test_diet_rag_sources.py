from ai_runtime.llm.rag.diet_sources import build_diet_rag_query, resolve_diet_rag_codes
from ai_runtime.llm.rag.keyword_retriever import retrieve_keyword_rag_contexts, retrieve_keyword_rag_matches
from ai_runtime.llm.rag.source_loader import load_all_rag_source_documents, load_rag_source_index


def test_rag_source_index_contains_ten_first_pass_documents() -> None:
    metadata = load_rag_source_index()

    assert len(metadata) == 10
    assert {item.disease_code for item in metadata} == {
        "HTN",
        "DM",
        "DL",
        "OBE",
        "FL",
        "CKD",
        "ANEM",
        "DIET_NUTRITION",
        "DIET_CAUTION",
        "DIET_FAQ",
    }


def test_only_enabled_rag_documents_are_loaded() -> None:
    documents = load_all_rag_source_documents()
    source_ids = {document.id for document in documents}

    assert {"hypertension", "diabetes", "dyslipidemia", "obesity", "diet_nutrition", "diet_caution"} <= source_ids
    assert "ckd" not in source_ids
    assert "anemia" not in source_ids
    assert "fatty_liver" not in source_ids
    assert "diet_faq" not in source_ids


def test_disabled_missing_sources_are_not_retrieved() -> None:
    contexts = retrieve_keyword_rag_contexts(
        user_message="신장기능 eGFR 요단백 식사 주의",
        disease_code="CKD",
        include_safety_disclaimer=False,
    )

    assert all(context.metadata["disease_code"] != "CKD" for context in contexts)


def test_htn_sodium_issue_maps_to_enabled_htn_and_diet_nutrition() -> None:
    codes = resolve_diet_rag_codes(issue_keys=["sodium_high"], disease_codes=["HTN"])
    query = build_diet_rag_query(issue_keys=["sodium_high"], disease_codes=["HTN"])
    contexts = retrieve_keyword_rag_contexts(user_message=query, disease_code="HTN", issue_keys=["sodium_high"])
    source_ids = {context.metadata["id"] for context in contexts}

    assert codes == ["HTN", "DIET_NUTRITION"]
    assert "hypertension" in source_ids
    assert "diet_nutrition" in source_ids


def test_ckd_kidney_caution_falls_back_to_diet_caution_when_ckd_disabled() -> None:
    codes = resolve_diet_rag_codes(issue_keys=["kidney_caution"], disease_codes=["CKD"])
    query = build_diet_rag_query(issue_keys=["kidney_caution"], disease_codes=["CKD"])
    contexts = retrieve_keyword_rag_contexts(user_message=query, issue_keys=["kidney_caution"])
    source_ids = {context.metadata["id"] for context in contexts}

    assert codes == ["DIET_CAUTION"]
    assert "ckd" not in source_ids
    assert "diet_caution" in source_ids


def test_keyword_rag_returns_empty_list_when_no_enabled_source_matches() -> None:
    matches = retrieve_keyword_rag_matches(
        user_message="전혀 관계없는 사용자 문장",
        disease_code="DIET_FAQ",
        include_safety_disclaimer=False,
    )

    assert matches == []
